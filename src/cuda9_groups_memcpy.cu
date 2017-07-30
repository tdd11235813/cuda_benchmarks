#include "cuda_helper.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cooperative_groups.h>
#include <iostream>
#include <stdexcept>
#include <limits>

using namespace cooperative_groups;

enum class MultiGrid {
  NO, YES
};

template<int TBlocksize, typename T>
__device__
void reduce(thread_group g, T *x) {

#pragma unroll
  for(int bs=1024; bs>1; bs=bs/2) {
    if( TBlocksize >= bs ) {
      if(g.thread_rank() < bs/2)
        x[g.thread_rank()] += x[g.thread_rank() + bs/2];
      g.sync();
    }
  }
}

template<int TBlocksize, typename TGroup, typename T>
__device__
T reduce(TGroup group, T *x, int n) {

  __shared__ T sdata[TBlocksize];

  // obtain default "current thread block" group
  thread_block my_block = this_thread_block();

  int lane = my_block.thread_rank(); // index \in {0,blocksize-1}

  //int i = 4 * blockIdx.x * TBlocksize + threadIdx.x;
  //int i = 4 * my_block.group_index().x * TBlocksize + lane;
  int i = 4 * blockIdx.x * TBlocksize + lane;

  sdata[lane] = 0;

  // --------
  // Level 1: [multi] group reduce
  // --------

  // reduce per thread with increased ILP by 4x unrolling sum.
  // the thread of our block reduces its 4 block-neighbored threads and advances by group-striding loop
  while (i+3*TBlocksize < n) {
    sdata[lane] += x[i] + x[i+TBlocksize] + x[i+2*TBlocksize] + x[i+3*TBlocksize];
    i += 4*group.size();
  }

  // doing the remaining blocks
  while(i<n) {
    sdata[lane] += x[i];
    i += group.size();
  }

  my_block.sync();

  // --------
  // Level 2: block + warp reduce
  // --------

  reduce<TBlocksize>(my_block, sdata);

  return sdata[0];
}

// TBlocksize must be power-of-2
template<int TBlocksize, typename T>
__global__
void kernel_reduce(T* x, T* y, int n)
{
  auto grid = this_grid();
  thread_block my_block = this_thread_block();

  T block_result = reduce<TBlocksize>(grid, x, n);

  // store block result to gmem
  if (my_block.thread_rank() == 0)
    y[blockIdx.x] = block_result;
//    y[my_block.group_index().x] = block_result;

  // grid synchronisation
  grid.sync();

  // --------
  // final reduce
  // - each block has written its result to gmem (data is coalesced)
  // - reduce the block results to the final value
  // - since we use coop kernels dynamic parallelism is not usable
  // --------

  // first block on first device
  if (blockIdx.x==0) {
    // reduce results of all the blocks stored in y
    T result = reduce<TBlocksize>(my_block, y, gridDim.x);
    // store result of reduction
    if(my_block.thread_rank() == 0)
      y[0] = result;
  }
}

template<int TBlocksize, typename T>
__global__
void kernel_reduce_multi(T* x, T* y, int n)
{
  auto grid = this_multi_grid(); // ! // cannot be used by cudaLaunchCooperativeKernel (will not terminate)
  thread_block my_block = this_thread_block();

  T block_result = reduce<TBlocksize>(grid, x, n);

  if (my_block.thread_rank() == 0)
    y[blockIdx.x + grid.grid_rank()*gridDim.x] = block_result; // !

  grid.sync();

  if (grid.grid_rank()==0 && blockIdx.x==0) { // !
    T result = reduce<TBlocksize>(my_block, y, grid.num_grids()*gridDim.x); // !
    if(my_block.thread_rank() == 0)
      y[0] = result;
  }
}

template<typename T, int TRuns, MultiGrid TMultiGrid>
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
  if(!prop.cooperativeLaunch)
    throw std::runtime_error("Device must support cooperativeLaunch property.");
  else
    std::cout << ", \"cooperativeLaunch\", \"supported\"";

  if(TMultiGrid == MultiGrid::YES) {
    if(!prop.cooperativeMultiDeviceLaunch)
      throw std::runtime_error("Device must support cooperativeMultiDeviceLaunch property.");
    else
      std::cout << ", \"cooperativeMultiDeviceLaunch\", \"supported\"";
  }
  std::cout << std::endl;

  const int nr_dev = 1;

  dim3 threads( 128 );
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
  dim3 blocks( 16*numSMs ); // factor must not exceed max number of active blocks per SM, otherwise runtime error will occur
  if( blocks.x > (n-1)/threads.x+1 )
    blocks.x = (n-1)/threads.x+1;

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

    // kernel<128><<<blocks, threads>>>(x, y, n);
    // NOTE: to use grid_groups we must use cudaLaunchCooperativeKernel
    // NOTE: such kernels cannot make use of dynamic parallelism

    if( TMultiGrid == MultiGrid::NO ) {
      CHECK_CUDA( cudaLaunchCooperativeKernel((const void*)(&kernel_reduce<128, T>),
                                              blocks, threads, args,
                                              (size_t)0/*smem*/,
                                              cstream) ); //(cudaStream_t)0/*stream*/) );
    } else {
      params[0].func = (void*)(&kernel_reduce_multi<128, T>);
      params[0].gridDim = blocks;
      params[0].blockDim = threads;
      params[0].args = args;
      params[0].sharedMem = 0;
      params[0].stream = cstream; // cannot use the NULL stream
      CHECK_CUDA(cudaLaunchCooperativeKernelMultiDevice(params, 1 /*numDevices*/));
    }
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
  std::cout << "[Single-Grid]\n\n";
  reduce<int,5, MultiGrid::NO>(1, 1<<26, 0);
  std::cout << "\n[Multi-Grid]\n\n";
  reduce<int,5, MultiGrid::YES>(1, 1<<26, 0);
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
