// http://stackoverflow.com/questions/24771730/the-efficiency-and-performance-of-ilp-for-the-nvidia-kepler-architecture
#include <stdio.h>
#include <time.h>

#define BLOCKSIZE 64

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b){
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

/************************************/
/* NO INSTRUCTION LEVEL PARALLELISM */
/************************************/
__global__ void ILP0(float* d_a, float* d_b, float* d_c) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  d_c[i] = d_a[i] + d_b[i];

}

/************************************/
/* INSTRUCTION LEVEL PARALLELISM X2 */
/************************************/
__global__ void ILP2(float* d_a, float* d_b, float* d_c) {

  // --- Loading the data
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  float ai = d_a[i];
  float bi = d_b[i];

  int stride = gridDim.x * blockDim.x;

  int j = i + stride;
  float aj = d_a[j];
  float bj = d_b[j];

  // --- Computing
  float ci = ai + bi;
  float cj = aj + bj;

  // --- Writing the data
  d_c[i] = ci;
  d_c[j] = cj;

}

/************************************/
/* INSTRUCTION LEVEL PARALLELISM X4 */
/************************************/
__global__ void ILP4(float* d_a, float* d_b, float* d_c) {

  // --- Loading the data
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  float ai = d_a[i];
  float bi = d_b[i];

  int stride = gridDim.x * blockDim.x;

  int j = i + stride;
  float aj = d_a[j];
  float bj = d_b[j];

  int k = j + stride;
  float ak = d_a[k];
  float bk = d_b[k];

  int l = k + stride;
  float al = d_a[l];
  float bl = d_b[l];

  // --- Computing
  float ci = ai + bi;
  float cj = aj + bj;
  float ck = ak + bk;
  float cl = al + bl;

  // --- Writing the data
  d_c[i] = ci;
  d_c[j] = cj;
  d_c[k] = ck;
  d_c[l] = cl;

}

/************************************/
/* INSTRUCTION LEVEL PARALLELISM X8 */
/************************************/
__global__ void ILP8(float* d_a, float* d_b, float* d_c) {

  // --- Loading the data
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  float ai = d_a[i];
  float bi = d_b[i];

  int stride = gridDim.x * blockDim.x;

  int j = i + stride;
  float aj = d_a[j];
  float bj = d_b[j];

  int k = j + stride;
  float ak = d_a[k];
  float bk = d_b[k];

  int l = k + stride;
  float al = d_a[l];
  float bl = d_b[l];

  int m = l + stride;
  float am = d_a[m];
  float bm = d_b[m];

  int n = m + stride;
  float an = d_a[n];
  float bn = d_b[n];

  int p = n + stride;
  float ap = d_a[p];
  float bp = d_b[p];

  int q = p + stride;
  float aq = d_a[q];
  float bq = d_b[q];

  // --- Computing
  float ci = ai + bi;
  float cj = aj + bj;
  float ck = ak + bk;
  float cl = al + bl;
  float cm = am + bm;
  float cn = an + bn;
  float cp = ap + bp;
  float cq = aq + bq;

  // --- Writing the data
  d_c[i] = ci;
  d_c[j] = cj;
  d_c[k] = ck;
  d_c[l] = cl;
  d_c[m] = cm;
  d_c[n] = cn;
  d_c[p] = cp;
  d_c[q] = cq;

}

/********/
/* MAIN */
/********/
int main(void) {

  float timing;
  cudaEvent_t start, stop;

  const int N = 65536*4; // --- ASSUMPTION: N can be divided by BLOCKSIZE

  float* a = (float*)malloc(N*sizeof(float));
  float* b = (float*)malloc(N*sizeof(float));
  float* c = (float*)malloc(N*sizeof(float));
  float* c_ref = (float*)malloc(N*sizeof(float));

  srand(time(NULL));
  for (int i=0; i<N; i++) {

    a[i] = rand() / RAND_MAX;
    b[i] = rand() / RAND_MAX;
    c_ref[i] = a[i] + b[i];

  }

  float* d_a; gpuErrchk(cudaMalloc((void**)&d_a,N*sizeof(float)));
  float* d_b; gpuErrchk(cudaMalloc((void**)&d_b,N*sizeof(float)));
  float* d_c0; gpuErrchk(cudaMalloc((void**)&d_c0,N*sizeof(float)));
  float* d_c2; gpuErrchk(cudaMalloc((void**)&d_c2,N*sizeof(float)));
  float* d_c4; gpuErrchk(cudaMalloc((void**)&d_c4,N*sizeof(float)));
  float* d_c8; gpuErrchk(cudaMalloc((void**)&d_c8,N*sizeof(float)));

  gpuErrchk(cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice));

  /******************/
  /* ILP0 TEST CASE */
  /******************/
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  ILP0<<<iDivUp(N,BLOCKSIZE),BLOCKSIZE>>>(d_a, d_b, d_c0);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timing, start, stop);
  printf("Elapsed time - ILP0:  %3.3f ms \n", timing);

  gpuErrchk(cudaMemcpy(c, d_c0, N*sizeof(float), cudaMemcpyDeviceToHost));

  // --- Checking the results
  for (int i=0; i<N; i++)
    if (c[i] != c_ref[i]) {

      printf("Error!\n");
      return 1;

    }

  printf("Test passed!\n");

  /******************/
  /* ILP2 TEST CASE */
  /******************/
  cudaEventRecord(start, 0);
  ILP2<<<(N/2)/BLOCKSIZE,BLOCKSIZE>>>(d_a, d_b, d_c2);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timing, start, stop);
  printf("Elapsed time - ILP2:  %3.3f ms \n", timing);

  gpuErrchk(cudaMemcpy(c, d_c2, N*sizeof(float), cudaMemcpyDeviceToHost));

  // --- Checking the results
  for (int i=0; i<N; i++)
    if (c[i] != c_ref[i]) {

      printf("Error!\n");
      return 1;

    }

  printf("Test passed!\n");

  /******************/
  /* ILP4 TEST CASE */
  /******************/
  cudaEventRecord(start, 0);
  ILP4<<<(N/4)/BLOCKSIZE,BLOCKSIZE>>>(d_a, d_b, d_c4);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timing, start, stop);
  printf("Elapsed time - ILP4:  %3.3f ms \n", timing);

  gpuErrchk(cudaMemcpy(c, d_c4, N*sizeof(float), cudaMemcpyDeviceToHost));

  // --- Checking the results
  for (int i=0; i<N; i++)
    if (c[i] != c_ref[i]) {

      printf("Error!\n");
      return 1;

    }

  printf("Test passed!\n");

  /******************/
  /* ILP8 TEST CASE */
  /******************/
  cudaEventRecord(start, 0);
  ILP8<<<(N/8)/BLOCKSIZE,BLOCKSIZE>>>(d_a, d_b, d_c8);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timing, start, stop);
  printf("Elapsed time - ILP8:  %3.3f ms \n", timing);

  gpuErrchk(cudaMemcpy(c, d_c8, N*sizeof(float), cudaMemcpyDeviceToHost));

  // --- Checking the results
  for (int i=0; i<N; i++)
    if (c[i] != c_ref[i]) {

      printf("%f %f\n",c[i],c_ref[i]);
      printf("Error!\n");
      return 1;

    }

  printf("Test passed!\n");
  return 0;
}
