// nvcc -o l052 lab05-reduction-solution-part2.cu -gencode arch=compute_35,code=sm_35
#include <iostream>
#include <cuda.h>
#include "cuda_helper.cuh"
#include "test_pagesize_include.cuh"

static constexpr int VERBOSE = 0;

struct TSettingsClass {
  static constexpr bool read_only = true;
  static constexpr unsigned iterations = 100000;
  static constexpr unsigned unroll_factor = 4;
};

template<typename TSettings, typename T>
__global__ void thread_gapping(T* const _data, unsigned _n, unsigned _ngap_thread, unsigned* _duration) {
	dev_fun<T> func;

  unsigned idx = blockDim.x*blockIdx.x+threadIdx.x;
  unsigned stride_data = _ngap_thread;
	unsigned index_data = idx*stride_data;

  if(index_data>=_n)
    return;

	unsigned offset = index_data;
	T temp = func.init(0);
  T * const _data_store_ptr = _data+offset+_n;

  unsigned int start_time, end_time;
  start_time = clock();
#pragma unroll (TSettings::unroll_factor)
	for(int j=0; j<TSettings::iterations; ++j) {
    const T v = func.load(_data, offset);
    if(TSettings::read_only) {
      offset ^= func.reduce(v); // assume zeros in _data so offset remains same
    } else {
      temp = v;
      func.store( _data_store_ptr, 0, temp );
    }
  }
  end_time = clock();
  _duration[idx] = end_time-start_time;

	if( offset != index_data ) // Does not occur
		_data[0] = func.init(offset);
}

// coalesced per warp, but stride among warps to have a gap between them of
// _ngap elements.
template<typename TSettings, typename T>
__global__ void warp_gapping(T* const _data, unsigned _n, unsigned _ngap_warp, unsigned* _duration) {
	dev_fun<T> func;

  unsigned idx = blockDim.x*blockIdx.x+threadIdx.x;
//  unsigned warps = gridDim.x*blockDim.x / 32;
  unsigned stride_data = _ngap_warp;
  // warp coalesced access
	unsigned index_data = idx/32*stride_data + (threadIdx.x&0x1f);

  if(index_data>=_n)
    return;

	unsigned offset = index_data;
	T temp = func.init(0);
  T * const _data_store_ptr = _data+offset+_n;

  // warmup
	for(int j=0; j<8; ++j) {
    const T v = func.load(_data, offset);
    if(TSettings::read_only) {
      offset ^= func.reduce(v); // assume zeros in _data so offset remains same
    } else {
      temp = v;
      func.store( _data_store_ptr, 0, temp );
    }
  }

  unsigned int start_time, end_time;
  start_time = clock();
#pragma unroll (TSettings::unroll_factor)
	for(int j=0; j<TSettings::iterations; ++j) {
    const T v = func.load(_data, offset);
    if(TSettings::read_only) {
      offset ^= func.reduce(v); // assume zeros in _data so offset remains same
    } else {
      temp = v;
      func.store( _data_store_ptr, 0, temp );
    }
  }
  end_time = clock();
  _duration[idx] = end_time-start_time;
	if( offset != index_data ) // Does not occur
		_data[0] = func.init(offset);
}

template<bool TWarpGapping, int TRuns, typename T>
void gapping(unsigned _block_dim, size_t _gap_in_bytes, int _nwarps) {
  cudaEvent_t cstart;
  cudaEvent_t cend;
  CHECK_CUDA(cudaEventCreate(&cstart));
  CHECK_CUDA(cudaEventCreate(&cend));
  size_t ngap = _gap_in_bytes/sizeof(T); // elements per gap
  size_t ngap_warp = ngap+32; // gap + warp
  size_t free=0, total=0;
  CHECK_CUDA( cudaMemGetInfo(&free, &total) );
  size_t n = (free/sizeof(T) / ngap_warp) * ngap_warp;
  unsigned nwarps = n/ngap_warp;
  if(_nwarps>0 && _nwarps<=nwarps) {
    nwarps = _nwarps;
    n = ngap_warp*nwarps;
  }else
    throw std::runtime_error("Invalid argument for number of warps.");

  T* data;
  unsigned* duration;
  unsigned blocks = (nwarps*32-1)/_block_dim+1;

  size_t bytes = n*sizeof(T);
  if(VERBOSE) {
    std::cout << "Free Memory on GPU: " << free/1048576 << " MiB"<<std::endl;
    std::cout << "Allocation on GPU:  " << bytes/1048576 << " MiB"<<std::endl;
    std::cout << "Byte per Element:   " << sizeof(T) << " B"<<std::endl;
    std::cout << "Number of Elements: " << n << std::endl;
    std::cout << "Number of Warps:    " << nwarps << std::endl;
  }
  CHECK_CUDA(cudaMalloc(&data, bytes));
  CHECK_CUDA(cudaMalloc(&duration, blocks*_block_dim*sizeof(unsigned)));
  CHECK_CUDA(cudaMemset(data, 0, bytes));
  CHECK_CUDA(cudaMemset(duration, 0, blocks*_block_dim*sizeof(unsigned)));

  warp_gapping<TSettingsClass><<<blocks, _block_dim>>>(data, n, ngap_warp, duration); // warmup
  CHECK_LAST("Kernel failed.");

  CHECK_CUDA(cudaEventRecord(cstart));
  for(int run=0; run<TRuns; ++run) {
    if(TWarpGapping)
      warp_gapping<TSettingsClass><<<blocks, _block_dim>>>(data, n, ngap_warp, duration);
    else
      thread_gapping<TSettingsClass><<<blocks, _block_dim>>>(data, n, ngap_warp/32, duration);
  }
  CHECK_CUDA(cudaEventRecord(cend));
  CHECK_CUDA(cudaEventSynchronize(cend));

  float kernel_time;
	CHECK_CUDA( cudaEventElapsedTime(&kernel_time, cstart, cend) );
	CHECK_CUDA( cudaEventDestroy(cstart) );
	CHECK_CUDA( cudaEventDestroy(cend) );
  CHECK_CUDA( cudaFree(data) );

  kernel_time/=TRuns;
  if(TWarpGapping) {
    std::cout << "\"warp gap [MiB]\", " << _gap_in_bytes/1048576
              << ", \"kernel [ms]\", " << kernel_time
              << ", \"runs\", " << TRuns
              << std::endl;
  } else {
    std::cout << "\"thread gap [Bytes]\", " << ngap_warp/32*sizeof(T)
              << ", \"kernel [ms]\", " << kernel_time
              << ", \"runs\", " << TRuns
              << std::endl;
  }
  unsigned* hduration = new unsigned[blocks*_block_dim];
  CHECK_CUDA(cudaMemcpy(hduration,duration,blocks*_block_dim*sizeof(unsigned),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(duration));
  for(int i=0; i<_block_dim; i+=32) {
    if(hduration[i])
      std::cout << " " << 1.0*hduration[i]/TSettingsClass::iterations;
  }
  std::cout<<std::endl;
  delete[] hduration;
}

/*
 * The main()-function.
 */
int main (int argc, char **argv)
{
  static constexpr int RUNS = 1;
  static constexpr size_t GAP_START = 1llu<<20;
  static constexpr size_t GAP_END = 1llu<<31;
  std::cout << listCudaDevices().str();
  unsigned block_dim = 128;

  for(size_t gap_in_bytes = GAP_START; gap_in_bytes<=GAP_END; gap_in_bytes <<= 1) {
    gapping<true,RUNS, int>(block_dim, gap_in_bytes, 2);
  }

  for(size_t gap_in_bytes = GAP_START; gap_in_bytes<=GAP_END; gap_in_bytes <<= 1) {
    gapping<false,RUNS, int>(block_dim, gap_in_bytes, 2);
  }


  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
