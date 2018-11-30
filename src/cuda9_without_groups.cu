#include "cuda_helper.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <limits>

// same number of registers are used when code is used in place
// (20 regs @GV100)
template<int TBlocksize, typename T>
__device__
void reduce_block(T *x, int n) {

  #pragma unroll
  for(int bs=TBlocksize,
        bsup=(TBlocksize+1)/2; // ceil(TBlocksize/2.0)
      bs>1;
      bs=bs/2,
        bsup=(bs+1)/2) // ceil(bs/2.0)
  {
    bool cond = threadIdx.x < bsup // only first half of block is working
               && (threadIdx.x+bsup) < TBlocksize // index for second half must be in bounds
               && (blockIdx.x*TBlocksize+threadIdx.x+bsup)<n; // if elem in second half has been initialized before
    if(cond)
    {
      x[threadIdx.x] += x[threadIdx.x + bsup];
    }
    __syncthreads();
  }
}

template<int TBlocksize, typename T>
__device__
T reduce(int tid, T *x, int n) {

  __shared__ T sdata[TBlocksize];

  int i = blockIdx.x * TBlocksize + tid;

  sdata[tid] = 0;

  // --------
  // Level 1: block reduce
  // --------

  // reduce per thread with increased ILP by 4x unrolling sum.
  // the thread of our block reduces its 4 grid-neighbored threads and advances by grid-striding loop
  while (i+3*gridDim.x*TBlocksize < n) {
    sdata[tid] += x[i] + x[i+gridDim.x*TBlocksize] + x[i+2*gridDim.x*TBlocksize] + x[i+3*gridDim.x*TBlocksize];
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
  reduce_block<TBlocksize>(sdata, n);
  return sdata[0];
}

template<int TBlocksize, typename T>
__global__
void kernel_reduce(T* x, T* y, int n)
{
  T block_result = reduce<TBlocksize>(threadIdx.x, x, n);

  // store block result to gmem
  if (threadIdx.x == 0)
    y[blockIdx.x] = block_result;
}

// TBlocksize must be power-of-2
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
  const int nbsm = 16;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
  dim3 blocks( nbsm*numSMs ); // factor must not exceed max number of active blocks per SM, otherwise runtime error will occur
//  dim3 blocks( 64*numSMs ); // factor must not exceed max number of active blocks per SM, otherwise runtime error will occur
//  dim3 blocks( (n-1)/TBlocksize+1 ); // factor must not exceed max number of active blocks per SM, otherwise runtime error will occur
  if( blocks.x > (n-1)/TBlocksize+1 )
    blocks.x = (n-1)/TBlocksize+1;

  std::cout << " #blocks/SM: "<< static_cast<float>(blocks.x)/numSMs << "\n"
            << " #blocks: " << blocks.x << "\n";

  T* h_x = new T[n];
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

  for(int r=0; r<TRuns; ++r) {
    CHECK_CUDA(cudaEventRecord(cstart, cstream));

    kernel_reduce<TBlocksize><<<blocks, TBlocksize, 0, cstream>>>(x, y, n);
    kernel_reduce<TBlocksize><<<1, TBlocksize, 0, cstream>>>(y, y, blocks.x);

    CHECK_CUDA( cudaEventRecord(cend, cstream) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaGetLastError() );
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

int main(int argc, const char** argv)
{
  static constexpr unsigned int REPETITIONS = 5;

  int dev=0;
  int n = 0;
  if(argc==2)
    n = atoi(argv[1]);
  if(n<2)
    n = 1<<28;
  reduce<int, REPETITIONS, 128>(1, n, dev);
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
