#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <vector>

#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <random>
#include <string>

#include <cuda_runtime.h>
//#include "cuda_helper.cuh"

#ifndef CUDA_DISABLE_ERROR_CHECKING
#define CHECK_CUDA(ans) check_cuda((ans), "", #ans, __FILE__, __LINE__)
#define CHECK_LAST(msg) check_cuda(cudaGetLastError(), msg, "CHECK_LAST", __FILE__, __LINE__)
#else
#define CHECK_CUDA(ans) {}
#define CHECK_LAST(msg) {}
#endif

inline
void throw_error(int code,
                 const char* error_string,
                 const char* msg,
                 const char* func,
                 const char* file,
                 int line) {
  throw std::runtime_error("CUDA error "
                           +std::string(msg)
                           +" "+std::string(error_string)
                           +" ["+std::to_string(code)+"]"
                           +" "+std::string(file)
                           +":"+std::to_string(line)
                           +" "+std::string(func)
    );
}
inline
void check_cuda(cudaError_t code, const char* msg, const char *func, const char *file, int line) {
  if (code != cudaSuccess) {
    throw_error(static_cast<int>(code),
                cudaGetErrorString(code), msg, func, file, line);
  }
}

using namespace std;

void DisplayHeader()
{
    int devCount;
    CHECK_CUDA(cudaGetDeviceCount(&devCount));
    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        CHECK_CUDA(cudaGetDeviceProperties(&props, i));
        cout << "# " << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
        cout << "#  MultiProcessors: " << props.multiProcessorCount << endl;
        cout << "#  Global memory:   " << props.totalGlobalMem / 1024 / 1024 << " MB" << endl;
    }
}

// take timings
static inline double getTimeInMS(){

    struct timeval starttime;
    gettimeofday(&starttime,0x0);
    
    return (double) starttime.tv_sec * (double) 1000 + (double) starttime.tv_usec / (double) 1000;
}


// --------------------- actual execution ------------------
// TODO L2 cache, L1 cache latencies

// SM with id!=0 try to disrupt TLB by loading with a large stride
/* parallel blocks on SMs have shown same SM_0 durations (no cache-miss effect seen)
different approach (assumes TLB L1 is not touched between kernels):
1) warm TLB in SM_0
2) refill TLB by other SM
3) try to use TLB from SM_0
 */
template<bool DISRUPT, bool GET_DURATION>
static __global__ void tlb_latency_with_disruptor(unsigned int * hashtable, unsigned hashtable_count, unsigned warmup, unsigned iterations, unsigned stride_count, unsigned offset, int smid0, int smxxx)
{
  extern __shared__ unsigned duration[]; // shared memory should be large enough to fill one SM

  unsigned smid;
  asm("mov.u32 %0, %smid;" : "=r"(smid) );

  if(!(DISRUPT || smid==smid0)) // only take 1st SM in non-disrupting mode
    return;
  if(DISRUPT && smid!=smxxx) // only SMxxx does run in disrupting mode
    return;
  if(threadIdx.x!=0)
    return;
  unsigned long start;
  unsigned int sum = 0;
	unsigned int pos = DISRUPT ? (stride_count*warmup + offset) % hashtable_count : offset;
  sum += pos; // ensure pos is set before entering loop
  for (unsigned int i = 0; i < iterations; i++){
    start = clock64();
    pos = hashtable[pos];
//    asm volatile ("ld.cg.u32 %0, [%1];" : "=r"(pos) : "l"(hashtable+pos));    // L2 only
//    asm volatile ("ld.ca.u32 %0, [%1];" : "=r"(pos) : "l"(hashtable+pos));    // L1/L2
    sum += pos; // ensure pos is set before taking clock
    duration[i] = static_cast<unsigned>(clock64()-start);
  }
  //printf("%d\n", smid);
  hashtable[hashtable_count+1] = sum;
  // in disrupting mode write SMxxx to dram
  if(DISRUPT) {
    hashtable[hashtable_count] = smid;
    return;
  }
  if(GET_DURATION && smid==smid0) { // only store durations one time
    unsigned poscas = atomicExch(hashtable+hashtable_count, 0xff); // to ensure only first thread on SM0 is writing to dram
    if(poscas==0xff) // durations already written so exit here
      return;
    if(poscas!=smxxx) { // if smxxx!=0, check if it was run on that smid in disrupting mode
      hashtable[hashtable_count+2+0] = 0xffffffff;
    } else {
      for (unsigned int i = 0; i < iterations; i++) {
        hashtable[hashtable_count+2+i] = duration[i];
      }
    }
  }
}


static __global__ void load_latency(unsigned int * hashtable, unsigned long hashtable_count, unsigned warmup, unsigned iterations)
{
  extern __shared__ unsigned duration[];
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int pos = threadID;
  unsigned long start;
  unsigned sum = 0;

  // warmup TLB
  for (unsigned int i = 0; i < warmup; i++) {
    pos = hashtable[pos];
  }
  sum = pos;
  pos = threadID + 128/sizeof(unsigned); // never loaded before to L1 (128b) and L2 cache (32b)
  sum += pos; // ensure pos is set before entering loop
  for (unsigned int i = 0; i < iterations; i++){
    start = clock64();
    pos = hashtable[pos];
//    asm volatile ("ld.cg.u32 %0, [%1];" : "=r"(pos) : "l"(hashtable+pos));    // L2 only
//    asm volatile ("ld.ca.u32 %0, [%1];" : "=r"(pos) : "l"(hashtable+pos));    // L1/L2
    sum += pos; // ensure pos is set before taking clock
    duration[i] = static_cast<unsigned>(clock64()-start);
  }
  hashtable[hashtable_count] = pos;

  hashtable[hashtable_count+1] = sum;
  for (unsigned int i = 0; i < iterations; i++){
    hashtable[hashtable_count+2+i] = duration[i];
  }
}

double runTest(unsigned blocks,unsigned threads,unsigned long hashtable_size_MB,unsigned access_width, unsigned stride, bool memset, unsigned warmup, unsigned iterations, const char* fname, int mode, int smid0, int smxxx){
	CHECK_CUDA(cudaDeviceReset());
  //CHECK_CUDA(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	unsigned int * hashtable;
  unsigned* hduration = new unsigned [iterations];//blocks*threads];
  size_t N = hashtable_size_MB*1048576llu/sizeof(unsigned int);
//  size_t bytes = hashtable_size_MB*1048576llu+(2llu)*sizeof(unsigned int);
  size_t bytes = hashtable_size_MB*1048576llu+(iterations+2llu)*sizeof(unsigned int);
	CHECK_CUDA(cudaMalloc(&hashtable,  bytes));
  if(memset)
    CHECK_CUDA(cudaMemset(hashtable, 0, bytes));

  unsigned int* hdata = new unsigned int[N+1];
//    memset(hdata, 0, N*sizeof(unsigned int));
/*    std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis(0, N-1);
      for(size_t i=0; i<N; ++i)
      hdata[i] = dis(gen);
*/

  /*for(size_t t=0; t<blocks*threads; ++t) {
    unsigned tv = t+1025;
    unsigned pos = t;
    for(int i=0;i<2*ITERATIONS; ++i) {
    hdata[pos] = ( tv/threads * pos + tv ) % N;
    pos = hdata[pos];
    }
    }
  */
  size_t stride_count = stride*1024llu/sizeof(unsigned);
  for(size_t t=0; t<N; ++t) {
    hdata[t] = ( t+stride_count ) % N;
  }
  hdata[N] = 0;

  if(access_width>1) {
    for(unsigned t=0; t<blocks*threads; t+=access_width) {
      for(unsigned a=1; a<access_width; ++a)
        hdata[t+a] = hdata[t];
    }
  }

  CHECK_CUDA(cudaMemcpy(hashtable, hdata, (N+1)*sizeof(unsigned), cudaMemcpyHostToDevice));
  delete[] hdata;



  CHECK_CUDA(cudaDeviceSynchronize());
	double start = getTimeInMS();

  if(mode==0)
    load_latency<<<blocks, threads, iterations*sizeof(unsigned)>>>(hashtable, N, warmup, iterations);
  else if(mode==1) {
    tlb_latency_with_disruptor<false, true><<<blocks, threads, iterations*sizeof(unsigned)>>>(hashtable, N, warmup, iterations, stride_count, 0, 0, 0);
  } else if(mode==2) {
    //warmup
    tlb_latency_with_disruptor<false, false><<<blocks, threads, iterations*sizeof(unsigned)>>>(hashtable, N, warmup, iterations, stride_count, 0, 0, 0);
    tlb_latency_with_disruptor<false, true><<<blocks, threads, iterations*sizeof(unsigned)>>>(hashtable, N, warmup, iterations, stride_count, 0, 0, 0);
  } else if(mode==3) {
    //warmup
    tlb_latency_with_disruptor<false, false><<<blocks, threads, iterations*sizeof(unsigned)>>>(hashtable, N, warmup, iterations, stride_count, 0, 0, 0);
    tlb_latency_with_disruptor<false, true><<<blocks, threads, iterations*sizeof(unsigned)>>>(hashtable, N, warmup, iterations, stride_count, 32, 0, 0);
  } else if(mode==4) {
    tlb_latency_with_disruptor<false, false><<<blocks, threads, iterations*sizeof(unsigned)>>>(hashtable, N, warmup, iterations, stride_count, 0, smid0, smxxx);
    tlb_latency_with_disruptor<true, false><<<blocks, threads, iterations*sizeof(unsigned)>>>(hashtable, N, warmup, iterations, stride_count, 0, smid0,smxxx);
    tlb_latency_with_disruptor<false, true><<<blocks, threads, iterations*sizeof(unsigned)>>>(hashtable, N, warmup, iterations, stride_count, 0, smid0,smxxx);
  } else if(mode==5) {
    tlb_latency_with_disruptor<false, false><<<blocks, threads, iterations*sizeof(unsigned)>>>(hashtable, N, warmup, iterations, stride_count, 0, smid0,smxxx);
    tlb_latency_with_disruptor<true, false><<<blocks, threads, iterations*sizeof(unsigned)>>>(hashtable, N, warmup, iterations, stride_count, 0, smid0,smxxx);
    tlb_latency_with_disruptor<false, true><<<blocks, threads, iterations*sizeof(unsigned)>>>(hashtable, N, warmup, iterations, stride_count, 32, smid0, smxxx); // 32-byte offset
  }
  CHECK_LAST( "Kernel failed." );
  CHECK_CUDA(cudaDeviceSynchronize());

	double stop = getTimeInMS();

  CHECK_CUDA(cudaMemcpy(hduration, hashtable+N+2, iterations*sizeof(unsigned), cudaMemcpyDeviceToHost));

  ofstream fdump;
  fdump.open(fname);
  double avgc=0;
  for(int b=0; b<iterations;++b) {
    avgc+=hduration[b];
    fdump<<b<<","<<hduration[b]<<endl;
  }
  fdump.close();
  avgc/=iterations;
	cout << "iterations, "<<iterations
       << ", size [MiB], " << hashtable_size_MB
       << ", stride [KiB], "<<stride
       << ", access_width, "<<access_width
       << ", warmup_steps, "<<warmup
       << ", blocks, "<<blocks<<", threads, "<<threads
       << ", time [ms], " << stop-start
       << ", avg cycles, "<<avgc//ITERATIONS
    ;
  cout << endl;

	CHECK_CUDA(cudaFree(hashtable));
  delete[] hduration;
	return stop-start;
}

int main(int argc, char **argv)
{
//	DisplayHeader();
	CHECK_CUDA(cudaSetDevice(0));

  int access_width = 1;
  int stride = 2048; // in KiB
  int start = 512;
  int end = 4*1024;
  int warmup = 256;
  int iterations = 256;
  std::string fname = "duration.csv";
  bool memset = false;
  if(argc>=2)
    access_width = atoi(argv[1]);
  if(argc>=4) {
    start = atoi(argv[2]);
    end = atoi(argv[3]);
  }
  if(argc>=5)
    stride = atoi(argv[4]);
  if(argc>=6)
    memset = atoi(argv[5])==1 ? true : false;
  if(argc>=7)
    warmup = atoi(argv[6]);
  if(argc>=8)
    iterations = atoi(argv[7]);
  if(argc>=9)
    fname = std::string(argv[8]);

  int mode = 0;
  if(argc>=10) {
    mode =atoi(argv[9]);
  }
  if(mode>0) {
    int numSMs; // for number of blocks
    CHECK_CUDA( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0) );
    int smid0=0;
    int smxxx=0;
    if(argc>=11)
      smid0=atoi(argv[10]);
    if(argc>=12)
      smxxx=atoi(argv[11]);
    runTest(2*numSMs,1,start,access_width,stride,memset,warmup,iterations,fname.c_str(),mode,smid0,smxxx);
  }else{
    for (double id = start; id <= end; id*=pow(2,0.1)) {
      unsigned int i = (unsigned(id+0.5)/2)*2;
      runTest(1,1,i,access_width,stride,memset,warmup,iterations,fname.c_str(),0,0,0);
    }
  }

  CHECK_CUDA(cudaDeviceReset());
	return 0;
}
