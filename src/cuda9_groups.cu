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

template<int TBlocksize, typename TGroup, typename T>
__device__
T reduce(TGroup group, T *x, int n) {

  __shared__ T sdata[TBlocksize];

  // obtain default "current thread block" group
  thread_block my_block = this_thread_block();

  int lane = my_block.thread_rank(); // index \in {0,blocksize-1}

  int i = blockIdx.x * TBlocksize + lane; // or: this_grid().thread_rank()

  sdata[lane] = x[i];
  i += group.size();

  // --------
  // Level 1: [multi] group reduce
  // --------

  // reduce per thread with increased ILP by 4x unrolling sum.
  // the thread of our block reduces its 4 grid-neighbored threads and advances by group-striding loop
  while (i+3*group.size() < n) {
    sdata[lane] += x[i] + x[i+group.size()] + x[i+2*group.size()] + x[i+3*group.size()];
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

#pragma unroll
  for(unsigned int bs=TBlocksize,
        bsup=(TBlocksize+1)/2; // ceil(TBlocksize/2.0)
      bs>1;
      bs=bs/2,
        bsup=(bs+1)/2) // ceil(bs/2.0)
  {
    bool cond = lane < bsup // only first half of block is working
               && (lane+bsup) < TBlocksize // index for second half must be in bounds
               && (this_grid().thread_rank()+bsup)<n; // if elem in second half has been initialized before
    if(cond)
    {
      sdata[lane] += sdata[lane + bsup];
    }
    my_block.sync();
  }
  return sdata[0];
}

template<int TBlocksize, typename T>
__global__
void kernel_reduce(T* x, T* y, int n)
{
  auto grid = this_grid();
  thread_block my_block = this_thread_block();

  if(grid.thread_rank()>=n)
    return;

  T block_result = reduce<TBlocksize>(grid, x, n);

  // store block result to gmem
  if (my_block.thread_rank() == 0)
    y[blockIdx.x] = block_result;
// or: y[my_block.group_index().x] = block_result;

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

  if(grid.thread_rank()>=n)
    return;

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
  int nbsm=0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbsm, kernel_reduce<128,T>, threads.x, 0);
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
  dim3 blocks( nbsm*numSMs ); // factor must not exceed max number of active blocks per SM, otherwise runtime error will occur
  if( blocks.x > (n-1)/threads.x+1 )
    blocks.x = (n-1)/threads.x+1;

  T* x;
  T* y;
  CHECK_CUDA( cudaMallocManaged(&x, n*sizeof(T)) );
  CHECK_CUDA( cudaMallocManaged(&y, nr_dev*blocks.x*sizeof(T)) );
  for (int i = 0; i < n; i++) {
    x[i] = init;
  }
  y[0] = 0;


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

  T result_gpu = y[0];

  std::cout << "Result (n = "<<n<<"):\n"
            << "GPU: " << result_gpu << " (min kernels time = "<< min_ms <<" ms)\n"
            << "expected: " << init*n <<"\n"
            << (init*n != result_gpu ? "MISMATCH!!" : "Success") << "\n"
            << "max bandwidth: "<<n*sizeof(T)/min_ms*1e-6<<" GB/s"
            << std::endl;


  CHECK_CUDA(cudaFree(x));
  CHECK_CUDA(cudaFree(y));
  CHECK_CUDA(cudaEventDestroy(cstart));
  CHECK_CUDA(cudaEventDestroy(cend));
  CHECK_CUDA(cudaStreamDestroy(cstream));

}

int main(void)
{
  std::cout << "[Single-Grid]\n\n";
  reduce<int,5, MultiGrid::NO>(1, 1<<28, 0);
  std::cout << "\n[Multi-Grid]\n\n";
  reduce<int,5, MultiGrid::YES>(1, 1<<28, 0);
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
