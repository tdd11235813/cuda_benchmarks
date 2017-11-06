#include "cuda_helper.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <limits>

/*
 * Reducing on warp level using implicit warp synchronisation.
 * (Not really recommended to rely on implicit warp synchronisation, but will be tackled with future CUDA >8)
 */
template <unsigned int TBlocksize, typename T>
inline
__device__ void warpReduce(volatile T *sdata, unsigned int tid) {
  if (TBlocksize >= 64) sdata[tid] += sdata[tid + 32];
  if (TBlocksize >= 32) sdata[tid] += sdata[tid + 16];
  if (TBlocksize >= 16) sdata[tid] += sdata[tid + 8];
  if (TBlocksize >= 8) sdata[tid] += sdata[tid + 4];
  if (TBlocksize >= 4) sdata[tid] += sdata[tid + 2];
  if (TBlocksize >= 2) sdata[tid] += sdata[tid + 1];
}

/*
 * Optimized reduction (addition) with GPU in shared memory.
 *
 * @param g_idata input (array in global GPU memory)
 * @param g_odata output (array in global GPU memory)
 * @param n number of elements to be processed
 */
template <unsigned int TBlocksize, typename T>
__global__ void
reduceAddGPUsmem(T *g_idata, T *g_odata, unsigned int n)
{
  __shared__ T sdata[TBlocksize];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*TBlocksize + tid;
  unsigned int gridSize = TBlocksize*gridDim.x;
  sdata[tid] = 0;
  // reduce per thread with increased ILP by 4x unrolling sum
  while (i+3*gridSize < n) {
    sdata[tid] += g_idata[i] + g_idata[i+gridSize] + g_idata[i+2*gridSize] + g_idata[i+3*gridSize];
    i += 4*gridSize;
  }
  // doing the rest
  while(i<n) {
    sdata[tid] += g_idata[i];
    i += gridSize;
  }

  __syncthreads();
  // block reduce
  if (TBlocksize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (TBlocksize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (TBlocksize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
  // warp reduce
  if (tid < 32)
    warpReduce<TBlocksize>(sdata, tid);
  // store block sum to gmem
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];

}

template<unsigned TBlocksize, typename T, int TRuns>
void runReduceAddGPUsmem(unsigned n)
{
  T *h_idata = new T[n];
  for (int i = 0; i < n; i++) {
    h_idata[i] = 1;
  }

  cudaEvent_t cstart, cend;
  // get number of streaming multiprocessors for our grid-striding loop
  int numSMs;
  int devId = 0;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);

  int numBlocks=min((n-1)/TBlocksize+1, 16*numSMs);

  /* allocate device memory and data */
  T *d_idata = NULL;
  T *d_blocksums = NULL;
  T *d_odata = NULL;
  CHECK_CUDA(cudaMalloc(&d_idata, n*sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_blocksums, numBlocks*sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_odata, sizeof(T)));
  CHECK_CUDA(cudaEventCreate(&cstart));
  CHECK_CUDA(cudaEventCreate(&cend));

  // copy data directly to device memory
  CHECK_CUDA(cudaMemcpy(d_idata, h_idata, n*sizeof(T), cudaMemcpyHostToDevice));

  T result_gpu=0;
  float milliseconds = 0;
  float min_ms = std::numeric_limits<float>::max();
  CHECK_CUDA(cudaMemcpy(&result_gpu, d_odata, sizeof(T), cudaMemcpyDeviceToHost));

  for(int run=0; run<TRuns; ++run) {
    CHECK_CUDA(cudaEventRecord(cstart));
    // Operations within the same stream are ordered (FIFO) and cannot overlap, so no explicit sync needed
    reduceAddGPUsmem<TBlocksize><<<numBlocks, TBlocksize>>>(d_idata, d_blocksums, n);
    reduceAddGPUsmem<TBlocksize><<<1, TBlocksize>>>(d_blocksums, d_odata, numBlocks);
    CHECK_CUDA(cudaEventRecord(cend));
    CHECK_CUDA(cudaEventSynchronize(cend));
    CHECK_CUDA( cudaGetLastError() );
    cudaEventElapsedTime(&milliseconds, cstart, cend);
    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  CHECK_CUDA(cudaMemcpy(&result_gpu, d_odata, sizeof(T), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_idata));
  CHECK_CUDA(cudaFree(d_odata));

  CHECK_CUDA(cudaEventDestroy(cstart));
  CHECK_CUDA(cudaEventDestroy(cend));

  std::cout << "Result (n = "<<n<<"):\n"
            << "GPU: " << result_gpu << " (min kernels time = "<< min_ms <<" ms)\n"
            << "expected: " << n <<"\n"
            << (n != result_gpu ? "MISMATCH!!" : "success") << "\n"
            << "max throughput: "<<n*sizeof(T)/min_ms*1e-6<<" GB/s"
            << std::endl;

  delete[] h_idata;
}

/*
 * The main()-function.
 */
int main (int argc, char **argv)
{
  unsigned n = 1<<28;
  runReduceAddGPUsmem<128, unsigned, 5>(n);

  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
