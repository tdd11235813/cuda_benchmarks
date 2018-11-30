#include <vector>
#include <iostream>
#include "cuda_helper.cuh"

static constexpr int ITERATIONS = 10;

template<typename T>
void perform_test(int n) {
  cudaStream_t stream0, stream1;
  CHECK_CUDA(cudaStreamCreate(&stream0));
  CHECK_CUDA(cudaStreamCreate(&stream1));

  size_t bytes = n*sizeof(T);
  std::cout << "n = " << n << " (" << 1.0*bytes/1048576 << " MB)\n";

  TimerCPU cpustart;
  double cpums;
  float gputime;
  cudaEvent_t start, end;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&end));

  T* ddata;
  CHECK_CUDA(cudaMalloc(&ddata, bytes));

  {
    // sync memcpy w vector.data()

    std::vector<T> data_in(n,1);
    std::vector<T> data_out(n,0);

    CHECK_CUDA(cudaEventRecord(start));
    cpustart.startTimer();

    for(int r=0; r<ITERATIONS; r++) {
      CHECK_CUDA(cudaMemcpy(ddata, data_in.data(), bytes, cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(data_out.data(), ddata, bytes, cudaMemcpyDeviceToHost));
    }

    cpums = cpustart.stopTimer();
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));

    CHECK_CUDA( cudaEventElapsedTime(&gputime, start, end) );
    std::cout << "Sync copy time on vector.data() = " << gputime/ITERATIONS << " ms (bandwidth = " << bytes/gputime*ITERATIONS*2e-6<<" GB/s)"
              << std::endl<<" [cpu call duration = "<<cpums/ITERATIONS<<" ms]"
              << std::endl;


    // cudaMemcpyAsync w vector.data()

    CHECK_CUDA(cudaEventRecord(start,stream0));
    cpustart.startTimer();

    for(int r=0; r<ITERATIONS; r++) {
      CHECK_CUDA(cudaMemcpyAsync(ddata, data_in.data(), bytes, cudaMemcpyHostToDevice, stream0));
      CHECK_CUDA(cudaMemcpyAsync(data_out.data(), ddata, bytes, cudaMemcpyDeviceToHost, stream1));
    }

    cpums = cpustart.stopTimer();
    CHECK_CUDA(cudaEventRecord(end,stream1));
    CHECK_CUDA(cudaEventSynchronize(end));

    CHECK_CUDA( cudaEventElapsedTime(&gputime, start, end) );
    std::cout << "Async copy time on vector.data() = " << gputime/ITERATIONS << " ms (bandwidth = " << bytes/gputime*ITERATIONS*2e-6<<" GB/s)"
              <<std::endl<< " [cpu call duration = "<<cpums/ITERATIONS<<" ms]"
              << std::endl;
  }


  // cudaMemcpyAsync w cudaMallocHost'd data
  T* data_pinned_in;
  T* data_pinned_out;
  CHECK_CUDA(cudaMallocHost(&data_pinned_in, bytes));
  CHECK_CUDA(cudaMallocHost(&data_pinned_out, bytes));

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaEventRecord(start,stream0));
  cpustart.startTimer();

  for(int r=0; r<ITERATIONS; r++) {
    CHECK_CUDA(cudaMemcpyAsync(ddata, data_pinned_in, bytes, cudaMemcpyHostToDevice, stream0));
    CHECK_CUDA(cudaMemcpyAsync(data_pinned_out, ddata, bytes, cudaMemcpyDeviceToHost, stream1));
  }

  cpums = cpustart.stopTimer();
  CHECK_CUDA(cudaEventRecord(end,stream1));
  CHECK_CUDA(cudaEventSynchronize(end));

	CHECK_CUDA( cudaEventElapsedTime(&gputime, start, end) );
  std::cout << "Async copy time (pinned) = " << gputime/ITERATIONS << " ms (bandwidth = " << bytes/gputime*ITERATIONS*2e-6<<" GB/s)"
            << std::endl<<" [cpu call duration = "<<cpums/ITERATIONS<<" ms]"
            << std::endl;

  CHECK_CUDA(cudaFree(ddata));
  CHECK_CUDA(cudaFreeHost(data_pinned_in));
  CHECK_CUDA(cudaFreeHost(data_pinned_out));
	CHECK_CUDA( cudaEventDestroy(start) );
	CHECK_CUDA( cudaEventDestroy(end) );
	CHECK_CUDA( cudaStreamDestroy(stream0) );
	CHECK_CUDA( cudaStreamDestroy(stream1) );

}

int main(int argc, char** argv)
{

  int n = 1<<20;
  int dev = 0;
  if(argc>=2)
    n = atoi(argv[1]);
  if(argc==3)
    dev = atoi(argv[2]);

	CHECK_CUDA(cudaSetDevice(dev));
  perform_test<int>(n);


	CHECK_CUDA(cudaDeviceReset());
	return 0;
}
