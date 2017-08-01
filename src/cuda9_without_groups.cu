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
  for(int bs=1024; bs>1; bs=bs/2) {
    if( TBlocksize >= bs ) {
      if(tid < bs/2)
        x[tid] += x[tid + bs/2];
      __syncthreads();
    }
  }
}

template<int TBlocksize, typename T>
__device__
T reduce(int tid, T *x, int n) {

  __shared__ T sdata[TBlocksize];

  //int i = 4 * blockIdx.x * TBlocksize + threadIdx.x;
  //int i = 4 * my_block.group_index().x * TBlocksize + lane;
  int i = 4 * blockIdx.x * TBlocksize + tid;

  sdata[tid] = 0;

  // --------
  // Level 1: block reduce
  // --------

  // reduce per thread with increased ILP by 4x unrolling sum.
  // the thread of our block reduces its 4 block-neighbored threads and advances by grid-striding loop
  while (i+3*TBlocksize < n) {
    sdata[tid] += x[i] + x[i+TBlocksize] + x[i+2*TBlocksize] + x[i+3*TBlocksize];
    i += 4*gridDim.x*TBlocksize;
  }

  // doing the remaining blocks
  while(i<n) {
    sdata[tid] += x[i];
    i += gridDim.x * TBlocksize;
  }

  __syncthreads();

  // --------
  // Level 2: block + warp reduce
  // --------

  reduce<TBlocksize>(tid, sdata);

  return sdata[0];
}

// TBlocksize must be power-of-2
template<int TBlocksize, typename T>
__global__
void kernel_reduce(T* x, T* y, int n)
{
  T block_result = reduce<TBlocksize>(threadIdx.x, x, n);

  // store block result to gmem
  if (threadIdx.x == 0)
    y[blockIdx.x] = block_result;
//    y[my_block.group_index().x] = block_result;
}


template<typename T, int TRuns, int TBlocksize>
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

  const int nr_dev = 1;

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
  dim3 blocks( 16*numSMs ); // factor must not exceed max number of active blocks per SM, otherwise runtime error will occur
  if( blocks.x > (n-1)/TBlocksize+1 )
    blocks.x = (n-1)/TBlocksize+1;

  T* h_x = new T[n];;
  T* x;
  T* y;
  CHECK_CUDA( cudaMalloc(&x, n*sizeof(T)) );
  CHECK_CUDA( cudaMalloc(&y, nr_dev*blocks.x*sizeof(T)) );
  for (int i = 0; i < n; i++) {
    h_x[i] = init;
  }
  CHECK_CUDA( cudaMemcpy( x, h_x, n*sizeof(T), cudaMemcpyHostToDevice) );


  float milliseconds = 0;
  float min_ms = std::numeric_limits<float>::max();
  cudaLaunchParams params[1];
  void* args[] = {(void*)&x, (void*)&y, (void*)&n};

  for(int r=0; r<TRuns; ++r) {
    CHECK_CUDA(cudaEventRecord(cstart, cstream));

    kernel_reduce<TBlocksize><<<blocks, TBlocksize, 0, cstream>>>(x, y, n);
    kernel_reduce<TBlocksize><<<1, TBlocksize, 0, cstream>>>(y, y, blocks.x);

    CHECK_CUDA( cudaEventRecord(cend, cstream) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  T result_gpu;
  CHECK_CUDA( cudaMemcpy( &result_gpu, y, sizeof(T), cudaMemcpyDeviceToHost) );

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

int main(void)
{
  int dev=0;
  reduce<int,5, 128>(1, 1<<26, dev);
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
