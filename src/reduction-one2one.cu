#include "cuda_helper.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <limits>

template<int TBlocksize, typename T>
__device__
void reduce(int tid, T *x) {

#pragma unroll
  for(int bs=TBlocksize, bsup=(TBlocksize+1)/2;
          bs>1;
          bs=bs/2, bsup=(bs+1)/2) {
    if(tid < bsup && tid+bsup<TBlocksize) {
      x[tid] += x[tid + bsup];
    }
    __syncthreads();
  }
}

template<int TBlocksize, typename T>
__device__
T reduce(int tid, T *x, int n) {

  __shared__ T sdata[TBlocksize];

  int i = blockIdx.x * TBlocksize + tid;

  // --------
  // Level 1: block reduce
  // --------

  // reduce per thread with increased ILP by 4x unrolling sum.
  // the thread of our block reduces its 4 grid-neighbored threads and advances by grid-striding loop
  sdata[tid] = x[i];

  __syncthreads();

  // --------
  // Level 2: block + warp reduce
  // --------

  reduce<TBlocksize>(tid, sdata);

  return sdata[0];
}

template<int TBlocksize, int TMaxWarpNum, typename T>
__global__
void kernel_reduce(T* x, T* y, int n)
{
  T block_result = reduce<TBlocksize>(threadIdx.x, x, n);

  unsigned warpid,smid;
  asm("mov.u32 %0, %%smid;":"=r"(smid));//get SM id
  asm("mov.u32 %0, %%warpid;":"=r"(warpid));//get warp id within SM

  // store block result to gmem
  if (threadIdx.x == 0)
    y[smid * TMaxWarpNum + warpid] += block_result;
}

template<int TBlocksize, typename T>
__global__
void kernel_reduce_2(T* x, T* y, int n)
{
  T block_result = reduce<TBlocksize>(threadIdx.x, x, n);

  // store block result to gmem
  if (threadIdx.x == 0)
    atomicAdd(y, block_result);
}

// TBlocksize must be power-of-2
template<typename T, int TRuns, int TBlocksize, int TMaxWarpNum>
void reduce(T init, size_t n, int dev) {

  CHECK_CUDA( cudaSetDevice(dev) );
  cudaDeviceProp prop;
  CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
  cudaEvent_t cstart, cend;
  CHECK_CUDA(cudaEventCreate(&cstart));
  CHECK_CUDA(cudaEventCreate(&cend));
  cudaStream_t cstream;
  CHECK_CUDA(cudaStreamCreate(&cstream));


  std::cout << getCUDADeviceInformations(dev).str();
  std::cout << std::endl;

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
  dim3 blocks = (n-1)/TBlocksize+1;
  dim3 blocks_2 = (TMaxWarpNum*numSMs-1)/TBlocksize+1;

  T* h_x = new T[n];
  T* x;
  T* y;
  T* z;
  CHECK_CUDA( cudaMalloc(&x, n*sizeof(T)) );
  CHECK_CUDA( cudaMalloc(&y, TMaxWarpNum*numSMs*sizeof(T)) );
  CHECK_CUDA( cudaMalloc(&z, sizeof(T)) );
  for (int i = 0; i < n; i++) {
    h_x[i] = init;
  }
  CHECK_CUDA( cudaMemcpy( x, h_x, n*sizeof(T), cudaMemcpyHostToDevice) );


  float milliseconds = 0;
  float min_ms = std::numeric_limits<float>::max();

  for(int r=0; r<TRuns; ++r) {
    CHECK_CUDA(cudaMemset(y, 0, TMaxWarpNum*numSMs*sizeof(T)));
    CHECK_CUDA(cudaMemset(z, 0, sizeof(T)));
    CHECK_CUDA(cudaEventRecord(cstart, cstream));

    kernel_reduce<TBlocksize, TMaxWarpNum><<<blocks, TBlocksize, 0, cstream>>>(x, y, n);
    kernel_reduce_2<TBlocksize><<<blocks_2, TBlocksize, 0, cstream>>>(y, z, blocks_2.x);

    CHECK_CUDA( cudaEventRecord(cend, cstream) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  T result_gpu;
  CHECK_CUDA( cudaMemcpy( &result_gpu, z, sizeof(T), cudaMemcpyDeviceToHost) );

  std::cout << "Result (n = "<<n<<"):\n"
            << "GPU: " << result_gpu << " (min kernels time = "<< min_ms <<" ms)\n"
            << "expected: " << init*n <<"\n"
            << (init*n != result_gpu ? "MISMATCH!!" : "Success") << "\n"
            << "max bandwidth: "<<n*sizeof(T)/min_ms*1e-6<<" GB/s"
            << std::endl;

  delete[] h_x;
  CHECK_CUDA(cudaFree(x));
  CHECK_CUDA(cudaFree(y));
  CHECK_CUDA(cudaEventDestroy(cstart));
  CHECK_CUDA(cudaEventDestroy(cend));
  CHECK_CUDA(cudaStreamDestroy(cstream));

}

int main(int argc, const char** argv)
{
  int dev=0;
  int n = 0;
  if(argc==2)
    n = atoi(argv[1]);
  if(n<2)
    n = 1<<28;
  reduce<int, 5, 128, 64>(1, n, dev);
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
