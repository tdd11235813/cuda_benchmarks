#include <stdio.h>
#include <stdint.h>
#include <random>
#include <iostream>
#include "cuda_helper.cuh"

static constexpr int ITERATIONS = 256;

__global__ void global_latency (unsigned int * my_array, int array_length, unsigned int * duration, unsigned int *index) {

	unsigned int start_time, end_time;
	unsigned int j = 0;

	__shared__ unsigned int s_tvalue[ITERATIONS];
	__shared__ unsigned int s_index[ITERATIONS];

	int k;

	for(k=0; k<ITERATIONS; k++){
		s_index[k] = 0;
		s_tvalue[k] = 0;
	}

	//first round, warm the TLB
	for (k = 0; k < ITERATIONS; k++)
		j = my_array[j];
//(my_array[j]+blockIdx.x*2048*1024)%array_length;

	//second round, begin timestamp
	for (k = 0; k < ITERATIONS; k++) {

		start_time = clock();

		j = my_array[j];
		s_index[k]= j;
		end_time = clock();
    s_tvalue[k] = end_time-start_time;

	}

//  if(blockIdx.x==0) {
    my_array[array_length] = j;
    my_array[array_length+1] = my_array[j];

    for(k=0; k<ITERATIONS; k++){
      index[k]= s_index[k];
      duration[k] = s_tvalue[k];
    }
//  }
}

__global__ void global_latency_funk (unsigned int * hashtable, unsigned int hashtable_size, unsigned int * duration) {
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int pos = threadID;

  // warmup
  for (unsigned int i = 0; i < ITERATIONS; i++){
    pos = hashtable[pos];
  }

  unsigned start = clock();
  for (unsigned int i = 0; i < ITERATIONS; i++){
    pos = hashtable[pos];
  }
  unsigned end = clock();
  duration[threadID] = end-start;
  hashtable[hashtable_size] = pos;
  hashtable[hashtable_size+1] = hashtable[pos];

}


void parametric_measure_global(size_t N, size_t stride) {
	CHECK_CUDA(cudaDeviceReset());

	size_t i;
	unsigned int * h_a;
	/* allocate arrays on CPU */
	h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N+2));
	unsigned int * d_a;
	/* allocate arrays on GPU */
	CHECK_CUDA(cudaMalloc(&d_a, sizeof(unsigned int) * (N+2)));

  /* initialize array elements on CPU with pointers into d_a. */
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, N-1);
	for (i = 0; i < N; i++) {
	//original:
		h_a[i] = (i+stride)%N;
//		h_a[i] = (i+32)%N;
//    h_a[i] = dis(gen);
	}

	h_a[N] = 0;
	h_a[N+1] = 0;
	/* copy array elements from CPU to GPU */

	unsigned int *h_index = (unsigned int *)malloc(sizeof(unsigned int)*ITERATIONS);
	unsigned int *h_timeinfo = (unsigned int *)malloc(sizeof(unsigned int)*ITERATIONS);

	unsigned int *duration;
	unsigned int *d_index;
	CHECK_CUDA(cudaMalloc (&duration, sizeof(unsigned int)*ITERATIONS));
	CHECK_CUDA(cudaMalloc( &d_index, sizeof(unsigned int)*ITERATIONS ));
  CHECK_CUDA(cudaMemset(duration,0, sizeof(unsigned int)*ITERATIONS));

//  CHECK_CUDA(cudaMemset(d_a, 0, sizeof(unsigned int) * (N+2)));
  CHECK_CUDA(cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice));
	/* launch kernel*/
	dim3 Db = dim3(1);
	dim3 Dg = dim3(1);


	CHECK_CUDA( cudaDeviceSynchronize() );
//	global_latency <<<Dg, Db>>>(d_a, N, duration, d_index);
	global_latency_funk <<<Dg, Db>>>(d_a, N, duration);
  CHECK_LAST( "Kernel failed." );
	CHECK_CUDA( cudaDeviceSynchronize() );


  CHECK_CUDA(cudaMemcpy(h_timeinfo, duration, sizeof(unsigned int)*ITERATIONS, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(h_index, d_index, sizeof(unsigned int)*ITERATIONS, cudaMemcpyDeviceToHost));

  double avgcyc=0.;
	for(i=0;i<ITERATIONS;i++) {
		//printf("%13d %6d\n", h_index[i], h_timeinfo[i]);
    avgcyc += h_timeinfo[i];
  }
  avgcyc /= ITERATIONS;
  printf("%d,%4.1f,%.3lf\n", ITERATIONS, sizeof(unsigned int)*(float)N/1024/1024, avgcyc);

	/* free memory on GPU */
	CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_index));
  CHECK_CUDA(cudaFree(duration));


  /*free memory on CPU */
  free(h_a);
  free(h_index);
	free(h_timeinfo);

//	CHECK_CUDA(cudaDeviceReset());
}

void measure_global(int stride_begin, int stride_end) {

	size_t N, stride;

	stride = 2048*1024/sizeof(unsigned int); //2MB stride
	//1. The L1 TLB has 16 entries. Test with N_min=28 *1024*256, N_max>32*1024*256
	//2. The L2 TLB has 65 entries. Test with N_min=128*1024*256, N_max=160*1024*256
	for (N = stride_begin*1024*(1024/sizeof(unsigned int)); N <= stride_end*1024*(1024/sizeof(unsigned int)); N+=stride) {
//		printf("\n=====%3.1f MB array, warm TLB, read 256 element====\n", sizeof(unsigned int)*(float)N/1024/1024);
//		printf("Stride = %d element, %d MB\n", stride, stride * sizeof(unsigned int)/1024/1024);
		parametric_measure_global(N, stride );
//		printf("===============================================\n\n");
	}
}


int main(int argc, char** argv)
{
  std::cout << listCudaDevices().str();
	CHECK_CUDA(cudaSetDevice(0));

  int stride_begin = 28; // in MiB
  int stride_end = 46;
  if(argc==3) {
    stride_begin = atoi(argv[1]);
    stride_end = atoi(argv[2]);
    measure_global(stride_begin, stride_end);
  }else {
    for(int sz = 16; sz<5000; sz<<=1)
      measure_global(sz, sz);
  }
	CHECK_CUDA(cudaDeviceReset());
	return 0;
}
