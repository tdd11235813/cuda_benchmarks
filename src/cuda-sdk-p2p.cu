/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
// 2019/10/31: Modified code from CUDA SDK p2pBandwidthLatencyTest to work with own helpers
#include "cuda_helper.cuh"
#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

const char *sSampleName = "P2P (Peer-to-Peer) GPU Bandwidth Latency Test";

typedef enum
{
    P2P_WRITE = 0,
    P2P_READ = 1,
}P2PDataTransfer;

typedef enum
{
    CE = 0,
    SM = 1,
}P2PEngine;

P2PEngine p2p_mechanism = CE; // By default use Copy Engine


__global__ void delay(volatile int *flag, unsigned long long timeout_clocks = 10000000)
{
    // Wait until the application notifies us that it has completed queuing up the
    // experiment, or timeout and exit, allowing the application to make progress
    long long int start_clock, sample_clock;
    start_clock = clock64();

    while (!*flag) {
        sample_clock = clock64();

        if (sample_clock - start_clock > timeout_clocks) {
            break;
        }
    }
}

// This kernel is for demonstration purposes only, not a performant kernel for p2p transfers.
__global__ void copyp2p(int4* __restrict__  dest, int4 const* __restrict__ src, size_t num_elems)
{
    size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;

    #pragma unroll(5)
    for (size_t i=globalId; i < num_elems; i+= gridSize)
    {
        dest[i] = src[i];
    }
}

///////////////////////////////////////////////////////////////////////////
//Print help screen
///////////////////////////////////////////////////////////////////////////
void printHelp(void)
{
    printf("Usage:  p2pBandwidthLatencyTest [OPTION]...\n");
    printf("Tests bandwidth/latency of GPU pairs using P2P and without P2P\n");
    printf("\n");

    printf("Options:\n");
    printf("h\t\tDisplay this help menu\n");
    printf("p\tUse P2P reads for data transfers between GPU pairs and show corresponding results.\n \t\tDefault used is P2P write operation.\n");
    printf("s\t\tUse SM intiated p2p transfers instead of Copy Engine\n");
}

void checkP2Paccess(int numGPUs)
{
    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));

        for (int j = 0; j < numGPUs; j++) {
            int access;
            if (i != j) {
                CHECK_CUDA(cudaDeviceCanAccessPeer(&access, i, j));
                printf("Device=%d %s Access Peer Device=%d\n", i, access ? "CAN" : "CANNOT", j);
            }
        }
    }
    printf("\n***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.\nSo you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.\n\n");
}

void performP2PCopy(int *dest, int destDevice, int *src, int srcDevice, int num_elems, int repeat, bool p2paccess, cudaStream_t streamToRun)
{
    int blockSize = 0;
    int numBlocks = 0;

    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, copyp2p));

    if (p2p_mechanism == SM && p2paccess)
    {
        for (int r = 0; r < repeat; r++) {
            copyp2p<<<numBlocks, blockSize, 0, streamToRun>>>((int4*)dest, (int4*)src, num_elems/4);
        }
    }
    else
    {
        for (int r = 0; r < repeat; r++) {
            cudaMemcpyPeerAsync(dest, destDevice, src, srcDevice, sizeof(int)*num_elems, streamToRun);
        }
    }
}

void outputBandwidthMatrix(int numGPUs, bool p2p, P2PDataTransfer p2p_method)
{
    int numElems = 10000000;
    int repeat = 5;
    volatile int *flag = NULL;
    vector<int *> buffers(numGPUs);
    vector<int *> buffersD2D(numGPUs); // buffer for D2D, that is, intra-GPU copy
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);
    vector<cudaStream_t> stream(numGPUs);

    CHECK_CUDA(cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable));

    for (int d = 0; d < numGPUs; d++) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
        CHECK_CUDA(cudaMalloc(&buffers[d], numElems * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&buffersD2D[d], numElems * sizeof(int)));
        CHECK_CUDA(cudaEventCreate(&start[d]));
        CHECK_CUDA(cudaEventCreate(&stop[d]));
    }

    vector<double> bandwidthMatrix(numGPUs * numGPUs);

    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));

        for (int j = 0; j < numGPUs; j++) {
            int access = 0;
            if (p2p) {
                CHECK_CUDA(cudaDeviceCanAccessPeer(&access, i, j));
                if (access) {
                    CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0 ));
                    CHECK_CUDA(cudaSetDevice(j));
                    CHECK_CUDA(cudaDeviceEnablePeerAccess(i, 0 ));
                    CHECK_CUDA(cudaSetDevice(i));
                }
            }

            CHECK_CUDA(cudaStreamSynchronize(stream[i]));

            // Block the stream until all the work is queued up
            // DANGER! - cudaMemcpy*Async may infinitely block waiting for
            // room to push the operation, so keep the number of repeatitions
            // relatively low.  Higher repeatitions will cause the delay kernel
            // to timeout and lead to unstable results.
            *flag = 0;
            delay<<< 1, 1, 0, stream[i]>>>(flag);
            CHECK_CUDA(cudaEventRecord(start[i], stream[i]));

            if (i == j) {
                // Perform intra-GPU, D2D copies
                performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, repeat, access, stream[i]);

            }
            else {
                if (p2p_method == P2P_WRITE)
                {
                    performP2PCopy(buffers[j], j, buffers[i], i, numElems, repeat, access, stream[i]);
                }
                else
                {
                    performP2PCopy(buffers[i], i, buffers[j], j, numElems, repeat, access, stream[i]);
                }
            }

            CHECK_CUDA(cudaEventRecord(stop[i], stream[i]));

            // Release the queued events
            *flag = 1;
            CHECK_CUDA(cudaStreamSynchronize(stream[i]));

            float time_ms;
            CHECK_CUDA(cudaEventElapsedTime(&time_ms, start[i], stop[i]));
            double time_s = time_ms / 1e3;

            double gb = numElems * sizeof(int) * repeat / (double)1e9;
            if (i == j) {
                gb *= 2;    //must count both the read and the write here
            }
            bandwidthMatrix[i * numGPUs + j] = gb / time_s;
            if (p2p && access) {
                CHECK_CUDA(cudaDeviceDisablePeerAccess(j));
                CHECK_CUDA(cudaSetDevice(j));
                CHECK_CUDA(cudaDeviceDisablePeerAccess(i));
                CHECK_CUDA(cudaSetDevice(i));
            }
        }
    }

    printf("   D\\D");

    for (int j = 0; j < numGPUs; j++) {
        printf("%6d ", j);
    }

    printf("\n");

    for (int i = 0; i < numGPUs; i++) {
        printf("%6d ", i);

        for (int j = 0; j < numGPUs; j++) {
            printf("%6.02f ", bandwidthMatrix[i * numGPUs + j]);
        }

        printf("\n");
    }

    for (int d = 0; d < numGPUs; d++) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaFree(buffers[d]));
        CHECK_CUDA(cudaFree(buffersD2D[d]));
        CHECK_CUDA(cudaEventDestroy(start[d]));
        CHECK_CUDA(cudaEventDestroy(stop[d]));
        CHECK_CUDA(cudaStreamDestroy(stream[d]));
    }

    CHECK_CUDA(cudaFreeHost((void *)flag));
}

void outputBidirectionalBandwidthMatrix(int numGPUs, bool p2p)
{
    int numElems = 10000000;
    int repeat = 5;
    volatile int *flag = NULL;
    vector<int *> buffers(numGPUs);
    vector<int *> buffersD2D(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);
    vector<cudaStream_t> stream0(numGPUs);
    vector<cudaStream_t> stream1(numGPUs);

    CHECK_CUDA(cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable));

    for (int d = 0; d < numGPUs; d++) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaMalloc(&buffers[d], numElems * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&buffersD2D[d], numElems * sizeof(int)));
        CHECK_CUDA(cudaEventCreate(&start[d]));
        CHECK_CUDA(cudaEventCreate(&stop[d]));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream0[d], cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream1[d], cudaStreamNonBlocking));
    }

    vector<double> bandwidthMatrix(numGPUs * numGPUs);

    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));

        for (int j = 0; j < numGPUs; j++) {
            int access = 0;
            if (p2p) {
                CHECK_CUDA(cudaDeviceCanAccessPeer(&access, i, j));
                if (access) {
                    CHECK_CUDA(cudaSetDevice(i)); // BUGFIX:
                    CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0 ));
                    CHECK_CUDA(cudaSetDevice(j));
                    CHECK_CUDA(cudaDeviceEnablePeerAccess(i, 0 ));
                    CHECK_CUDA(cudaSetDevice(i));
                }
            }


            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaStreamSynchronize(stream0[i]));
            CHECK_CUDA(cudaStreamSynchronize(stream1[j]));

            // Block the stream until all the work is queued up
            // DANGER! - cudaMemcpy*Async may infinitely block waiting for
            // room to push the operation, so keep the number of repeatitions
            // relatively low.  Higher repeatitions will cause the delay kernel
            // to timeout and lead to unstable results.
            *flag = 0;
            CHECK_CUDA(cudaSetDevice(i));
            // No need to block stream1 since it'll be blocked on stream0's event
            delay<<< 1, 1, 0, stream0[i]>>>(flag);

            // Force stream1 not to start until stream0 does, in order to ensure
            // the events on stream0 fully encompass the time needed for all operations
            CHECK_CUDA(cudaEventRecord(start[i], stream0[i]));
            CHECK_CUDA(cudaStreamWaitEvent(stream1[j], start[i], 0));

            if (i == j) {
                // For intra-GPU perform 2 memcopies buffersD2D <-> buffers
                performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, repeat, access, stream0[i]);
                performP2PCopy(buffersD2D[i], i, buffers[i], i, numElems, repeat, access, stream1[i]);
            }
            else {
                if (access && p2p_mechanism == SM)
                {
                    CHECK_CUDA(cudaSetDevice(j));
                }
                performP2PCopy(buffers[i], i, buffers[j], j, numElems, repeat, access, stream1[j]);
                if (access && p2p_mechanism == SM)
                {
                    CHECK_CUDA(cudaSetDevice(i));
                }
                performP2PCopy(buffers[j], j, buffers[i], i, numElems, repeat, access, stream0[i]);
            }

            // Notify stream0 that stream1 is complete and record the time of
            // the total transaction
            CHECK_CUDA(cudaEventRecord(stop[j], stream1[j]));
            CHECK_CUDA(cudaStreamWaitEvent(stream0[i], stop[j], 0));
            CHECK_CUDA(cudaEventRecord(stop[i], stream0[i]));

            // Release the queued operations
            *flag = 1;
            CHECK_CUDA(cudaStreamSynchronize(stream0[i]));
            CHECK_CUDA(cudaStreamSynchronize(stream1[j]));

            float time_ms;
            CHECK_CUDA(cudaEventElapsedTime(&time_ms, start[i], stop[i]));
            double time_s = time_ms / 1e3;

            double gb = 2.0 * numElems * sizeof(int) * repeat / (double)1e9;
            if (i == j) {
                gb *= 2;    //must count both the read and the write here
            }
            bandwidthMatrix[i * numGPUs + j] = gb / time_s;
            if (p2p && access) {
                CHECK_CUDA(cudaSetDevice(i));
                CHECK_CUDA(cudaDeviceDisablePeerAccess(j));
                CHECK_CUDA(cudaSetDevice(j));
                CHECK_CUDA(cudaDeviceDisablePeerAccess(i));
            }
        }
    }

    printf("   D\\D");

    for (int j = 0; j < numGPUs; j++) {
        printf("%6d ", j);
    }

    printf("\n");

    for (int i = 0; i < numGPUs; i++) {
        printf("%6d ", i);

        for (int j = 0; j < numGPUs; j++) {
            printf("%6.02f ", bandwidthMatrix[i * numGPUs + j]);
        }

        printf("\n");
    }

    for (int d = 0; d < numGPUs; d++) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaFree(buffers[d]));
        CHECK_CUDA(cudaFree(buffersD2D[d]));
        CHECK_CUDA(cudaEventDestroy(start[d]));
        CHECK_CUDA(cudaEventDestroy(stop[d]));
        CHECK_CUDA(cudaStreamDestroy(stream0[d]));
        CHECK_CUDA(cudaStreamDestroy(stream1[d]));
    }

    CHECK_CUDA(cudaFreeHost((void *)flag));
}

void outputLatencyMatrix(int numGPUs, bool p2p, P2PDataTransfer p2p_method)
{
    int repeat = 100;
    volatile int *flag = NULL;
    vector<int *> buffers(numGPUs);
    vector<int *> buffersD2D(numGPUs);  // buffer for D2D, that is, intra-GPU copy
    vector<cudaStream_t> stream(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);

    CHECK_CUDA(cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable));

    TimerCPU timer;;

    for (int d = 0; d < numGPUs; d++) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
        CHECK_CUDA(cudaMalloc(&buffers[d], sizeof(int)));
        CHECK_CUDA(cudaMalloc(&buffersD2D[d], sizeof(int)));
        CHECK_CUDA(cudaEventCreate(&start[d]));
        CHECK_CUDA(cudaEventCreate(&stop[d]));
    }

    vector<double> gpuLatencyMatrix(numGPUs * numGPUs);
    vector<double> cpuLatencyMatrix(numGPUs * numGPUs);

    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));

        for (int j = 0; j < numGPUs; j++) {
            int access = 0;
            if (p2p) {
                CHECK_CUDA(cudaDeviceCanAccessPeer(&access, i, j));
                if (access) {
                    CHECK_CUDA(cudaSetDevice(i)); // BUGFIX:
                    CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0));
                    CHECK_CUDA(cudaSetDevice(j));
                    CHECK_CUDA(cudaDeviceEnablePeerAccess(i, 0));
                    CHECK_CUDA(cudaSetDevice(i));
                }
            }
            CHECK_CUDA(cudaStreamSynchronize(stream[i]));

            // Block the stream until all the work is queued up
            // DANGER! - cudaMemcpy*Async may infinitely block waiting for
            // room to push the operation, so keep the number of repeatitions
            // relatively low.  Higher repeatitions will cause the delay kernel
            // to timeout and lead to unstable results.
            *flag = 0;
            delay<<< 1, 1, 0, stream[i]>>>(flag);
            CHECK_CUDA(cudaEventRecord(start[i], stream[i]));

            timer.startTimer();
            if (i == j) {
                // Perform intra-GPU, D2D copies
                performP2PCopy(buffers[i], i, buffersD2D[i], i, 1, repeat, access, stream[i]);
            }
            else {
                if (p2p_method == P2P_WRITE)
                {
                    performP2PCopy(buffers[j], j, buffers[i], i, 1, repeat, access, stream[i]);
                }
                else
                {
                    performP2PCopy(buffers[i], i, buffers[j], j, 1, repeat, access, stream[i]);
                }
            }
            double cpu_time_ms = timer.stopTimer();;

            CHECK_CUDA(cudaEventRecord(stop[i], stream[i]));
            // Now that the work has been queued up, release the stream
            *flag = 1;
            CHECK_CUDA(cudaStreamSynchronize(stream[i]));

            float gpu_time_ms;
            CHECK_CUDA(cudaEventElapsedTime(&gpu_time_ms, start[i], stop[i]));

            gpuLatencyMatrix[i * numGPUs + j] = gpu_time_ms * 1e3 / repeat;
            cpuLatencyMatrix[i * numGPUs + j] = cpu_time_ms * 1e3 / repeat;
            if (p2p && access) {
                CHECK_CUDA(cudaDeviceDisablePeerAccess(j));
                CHECK_CUDA(cudaSetDevice(j));
                CHECK_CUDA(cudaDeviceDisablePeerAccess(i));
                CHECK_CUDA(cudaSetDevice(i));
            }
        }
    }

    printf("   GPU");

    for (int j = 0; j < numGPUs; j++) {
        printf("%6d ", j);
    }

    printf("\n");

    for (int i = 0; i < numGPUs; i++) {
        printf("%6d ", i);

        for (int j = 0; j < numGPUs; j++) {
            printf("%6.02f ", gpuLatencyMatrix[i * numGPUs + j]);
        }

        printf("\n");
    }

    printf("\n   CPU");

    for (int j = 0; j < numGPUs; j++) {
        printf("%6d ", j);
    }

    printf("\n");

    for (int i = 0; i < numGPUs; i++) {
        printf("%6d ", i);

        for (int j = 0; j < numGPUs; j++) {
            printf("%6.02f ", cpuLatencyMatrix[i * numGPUs + j]);
        }

        printf("\n");
    }

    for (int d = 0; d < numGPUs; d++) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaFree(buffers[d]));
        CHECK_CUDA(cudaFree(buffersD2D[d]));
        CHECK_CUDA(cudaEventDestroy(start[d]));
        CHECK_CUDA(cudaEventDestroy(stop[d]));
        CHECK_CUDA(cudaStreamDestroy(stream[d]));
    }

    CHECK_CUDA(cudaFreeHost((void *)flag));
}

int main(int argc, char **argv)
{
    int numGPUs;
    P2PDataTransfer p2p_method = P2P_WRITE;

    CHECK_CUDA(cudaGetDeviceCount(&numGPUs));

    //process command line args
    if (argc>1 && argv[1][0]=='h')
    {
        printHelp();
        return 0;
    }

    if (argc>1 && argv[1][0]=='p')
    {
        p2p_method = P2P_READ;
    }

    if (argc>1 && argv[1][0]=='s')
    {
        p2p_mechanism = SM;
    }

    printf("[%s]\n", sSampleName);

    //output devices
    for (int i = 0; i < numGPUs; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        printf("Device: %d, %s, pciBusID: %x, pciDeviceID: %x, pciDomainID:%x\n", i, prop.name, prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
    }

    //checkP2Paccess(numGPUs);

    //Check peer-to-peer connectivity
    printf("P2P Connectivity Matrix\n");
    printf("     D\\D");

    for (int j = 0; j < numGPUs; j++) {
        printf("%6d", j);
    }
    printf("\n");

    for (int i = 0; i < numGPUs; i++) {
        printf("%6d\t", i);
        for (int j = 0; j < numGPUs; j++) {
            if (i != j) {
                int access;
                CHECK_CUDA(cudaDeviceCanAccessPeer(&access, i, j));
                printf("%6d", (access) ? 1 : 0);
            }
            else {
                printf("%6d", 1);
            }
        }
        printf("\n");
    }

    printf("Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)\n");
    outputBandwidthMatrix(numGPUs, false, P2P_WRITE);
    printf("Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)\n");
    outputBandwidthMatrix(numGPUs, true, P2P_WRITE);
    if (p2p_method == P2P_READ)
    {
        printf("Unidirectional P2P=Enabled Bandwidth (P2P Reads) Matrix (GB/s)\n");
        outputBandwidthMatrix(numGPUs, true, p2p_method);
    }
    printf("Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)\n");
    outputBidirectionalBandwidthMatrix(numGPUs, false);
    printf("Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)\n");
    outputBidirectionalBandwidthMatrix(numGPUs, true);

    printf("P2P=Disabled Latency Matrix (us)\n");
    outputLatencyMatrix(numGPUs, false, P2P_WRITE);
    printf("P2P=Enabled Latency (P2P Writes) Matrix (us)\n");
    outputLatencyMatrix(numGPUs, true, P2P_WRITE);
    if (p2p_method == P2P_READ)
    {
        printf("P2P=Enabled Latency (P2P Reads) Matrix (us)\n");
        outputLatencyMatrix(numGPUs, true, p2p_method);
    }

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    exit(EXIT_SUCCESS);
}
