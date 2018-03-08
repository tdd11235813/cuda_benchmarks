#include "cuda_helper.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <vector>

void saxpy(int n, float a, float *x, float *y)
{
    for (int i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];
}

template<typename T>
__global__
void saxpy_gridStride(int n, T a, T *x, T *y) {

 int i = blockIdx.x * blockDim.x + threadIdx.x;
const int s = blockDim.x * gridDim.x;
	while( i+s*4 <= n ) 
	{
		y[i] = a * x[i] + y[i];
		i += s;
		y[i] = a * x[i] + y[i];
		i += s;
		y[i] = a * x[i] + y[i];
		i += s;
		y[i] = a * x[i] + y[i];
		i += s;
	}
	while(	i < n  	) 
	{
		y[i] = a * x[i] + y[i];
		i += s;
	}
}

template<typename T>
__global__
void saxpy_gridUnroll(int n, T a, T *x, T *y) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) 
		y[i] = a * x[i] + y[i];
}

// TBlocksize must be power-of-2
template<typename T, int TRuns, int TBlocksize>
void run_stride(T init, int n, int dev) {

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
  dim3 blocks( 1024/TBlocksize*numSMs*2*2 );
	std::cout << "\trunning <<<" << blocks.x << ", " << TBlocksize << std::endl;

  std::vector<T> h_x(n), h_y(n), h_z(n);
  T* x;
  T* y;
  CHECK_CUDA( cudaMalloc(&x, n*sizeof(T)) );
  CHECK_CUDA( cudaMalloc(&y, n*sizeof(T)) );
  for (int i = 0; i < n; i++) {
    h_x[i] = init;
    h_y[i] = init;
    h_z[i] = init;
  }
	saxpy(n, init, h_x.data(), h_z.data());
std::cerr << h_x[0] << ' ' << h_y[0] << '\n';
  CHECK_CUDA( cudaMemcpy( x, h_x.data(), n*sizeof(T), cudaMemcpyHostToDevice) );


  float milliseconds = 0;
  float min_ms = std::numeric_limits<float>::max();

  for(int r=0; r<TRuns; ++r) {
	  CHECK_CUDA( cudaMemcpy( y, h_y.data(), n*sizeof(T), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaEventRecord(cstart, cstream));

    saxpy_gridStride<T><<<blocks,TBlocksize, 0, cstream>>>(n, init, x, y);

    CHECK_CUDA( cudaEventRecord(cend, cstream) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  CHECK_CUDA( cudaMemcpy( h_y.data(), y, n*sizeof(T), cudaMemcpyDeviceToHost) );
  for (int i = 0; i < n; i++) {
		if(h_y[i] != h_z[i]) {
			std::cout << "elem " << i << " does not match: " << h_y[i] << " vs " << h_z[i] << std::endl;
			return;
		}
	}

  std::cout << "Result saxpy_gridStride (n = "<<n<<"):\n"
            << "\tGPU: " << " (min kernels time = "<< min_ms <<" ms)\n"
            << "\tmax bandwidth: "<<n*sizeof(T)/min_ms*1e-6<<" GB/s"
            << std::endl;

  CHECK_CUDA(cudaFree(x));
  CHECK_CUDA(cudaFree(y));
  CHECK_CUDA(cudaEventDestroy(cstart));
  CHECK_CUDA(cudaEventDestroy(cend));
  CHECK_CUDA(cudaStreamDestroy(cstream));

}

// TBlocksize must be power-of-2
template<typename T, int TRuns, int TBlocksize>
void run_unroll(T init, int n, int dev) {

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
  dim3 blocks( n/TBlocksize);
	std::cout << "\trunning <<<" << blocks.x << ", " << TBlocksize << std::endl;

  std::vector<T> h_x(n), h_y(n);
  T* x;
  T* y;
  CHECK_CUDA( cudaMalloc(&x, n*sizeof(T)) );
  CHECK_CUDA( cudaMalloc(&y, n*sizeof(T)) );
  for (int i = 0; i < n; i++) {
    h_x[i] = init;
    h_y[i] = init;
  }
  CHECK_CUDA( cudaMemcpy( x, h_x.data(), n*sizeof(T), cudaMemcpyHostToDevice) );
  CHECK_CUDA( cudaMemcpy( y, h_y.data(), n*sizeof(T), cudaMemcpyHostToDevice) );


  float milliseconds = 0;
  float min_ms = std::numeric_limits<float>::max();

  for(int r=0; r<TRuns; ++r) {
    CHECK_CUDA(cudaEventRecord(cstart, cstream));

    saxpy_gridUnroll<T><<<blocks,TBlocksize, 0, cstream>>>(n, init, x, y);

    CHECK_CUDA( cudaEventRecord(cend, cstream) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  CHECK_CUDA( cudaMemcpy( h_y.data(), y, n*sizeof(T), cudaMemcpyDeviceToHost) );

  std::cout << "Result saxpy_gridUnroll (n = "<<n<<"):\n"
            << "\tGPU: " << " (min kernels time = "<< min_ms <<" ms)\n"
            << "\tmax bandwidth: "<<n*sizeof(T)/min_ms*1e-6<<" GB/s"
            << std::endl;

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
  const float init = 2.;
  run_stride<float, 5, 512>(init, n, dev);
  run_unroll<float, 5, 512>(init, n, dev);
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
