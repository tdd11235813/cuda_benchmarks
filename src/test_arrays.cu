/*
    AoS tends to be more readable to the programmer as each "object" is kept together.
    AoS may have better cache locality if all the members of the struct are accessed together.
    SoA could potentially be more efficient since grouping same datatypes together sometimes exposes vectorization.
    In many cases SoA uses less memory because padding is only between arrays rather than between every struct.
http://stackoverflow.com/questions/8377667/layout-in-memory-of-a-struct-struct-of-arrays-and-array-of-structs-in-c-c/8377717#8377717

Tesla K20Xm with peak BW = 249.6
AoS [int]     , 100 runs, 1048576 objects, 0.195823 ms [avg], 171.35 GB/s [avg]
flat-AoS [int], 100 runs, 1048576 objects, 0.282051 ms [avg], 118.966 GB/s [avg]
flat-SoA [int], 100 runs, 1048576 objects, 0.208467 ms [avg], 160.958 GB/s [avg]
SoA [int]     , 100 runs, 1048576 objects, 0.208764 ms [avg], 160.729 GB/s [avg]
AoS [float]     , 100 runs, 1048576 objects, 0.196342 ms [avg], 170.898 GB/s [avg]
flat-AoS [float], 100 runs, 1048576 objects, 0.281161 ms [avg], 119.342 GB/s [avg]
flat-SoA [float], 100 runs, 1048576 objects, 0.208716 ms [avg], 160.766 GB/s [avg]
SoA [float]     , 100 runs, 1048576 objects, 0.208318 ms [avg], 161.073 GB/s [avg]

Tesla K80 with peak BW = 240.48
AoS [int]     , 100 runs, 1048576 objects, 0.196408 ms [avg], 170.841 GB/s [avg]
flat-AoS [int], 100 runs, 1048576 objects, 0.278536 ms [avg], 120.467 GB/s [avg]
flat-SoA [int], 100 runs, 1048576 objects, 0.201841 ms [avg], 166.242 GB/s [avg]
SoA [int]     , 100 runs, 1048576 objects, 0.202035 ms [avg], 166.082 GB/s [avg]
AoS [float]     , 100 runs, 1048576 objects, 0.196226 ms [avg], 170.999 GB/s [avg]
flat-AoS [float], 100 runs, 1048576 objects, 0.278812 ms [avg], 120.348 GB/s [avg]
flat-SoA [float], 100 runs, 1048576 objects, 0.201781 ms [avg], 166.291 GB/s [avg]
SoA [float]     , 100 runs, 1048576 objects, 0.201866 ms [avg], 166.222 GB/s [avg]

result: 70~75% of peak bw due to ECC and hardware inefficiencies - almost equal to D2D copy
*/

#include <iostream>
#include <vector>
#include <vector_types.h>
#include <type_traits>
#include <stdexcept>
using namespace std;

#define CHECK_CUDA(ans) check_cuda((ans), "", #ans, __FILE__, __LINE__)
#define CHECK_LAST(msg) check_cuda(cudaGetLastError(), msg, "CHECK_LAST", __FILE__, __LINE__)

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

// ___________________________________________________________________

template<typename T>
struct SoA4 {
  using type = T;
  T* x;
  T* y;
  T* z;
  T* w;
};

cudaEvent_t cstart, cstop;

// ____________ kernel _______________________________________________________


template<bool TFlat, bool TSoA, typename T, typename std::enable_if<TFlat && TSoA,bool>::type = 0>
__global__
void dkernel(T* values, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x)
  {
    T v1 = values[i];
    T v2 = values[i+n];
    T v3 = values[i+2*n];
    T v4 = values[i+3*n];
    values[i] = v4;
    values[i+n] = v3;
    values[i+2*n] = v2;
    values[i+3*n] = v1;
  }
}

template<bool TFlat, bool TSoA, typename T, typename std::enable_if<TFlat && !TSoA,bool>::type = 0>
__global__
void dkernel(T* values, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x)
  {
    int s = 4*i;
    T v1 = values[s];
    T v2 = values[s+1];
    T v3 = values[s+2];
    T v4 = values[s+3];
    values[s] = v4;
    values[s+1] = v3;
    values[s+2] = v2;
    values[s+3] = v1;
  }
}
/* // fast as int4
template<>
__global__
void dkernel<true,false,int>(int* values, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x)
  {
    int4 v = reinterpret_cast<int4*>(values)[i];
    int4 *vp = reinterpret_cast<int4*>(values)+i;
    vp->x = v.x;
    vp->y = v.y;
    vp->z = v.z;
    vp->w = v.w;
  }
}*/

template<bool TFlat, bool TSoA, typename T, typename std::enable_if<!TFlat && !TSoA,bool>::type = 0>
__global__
void dkernel(T* values, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x)
  {
    auto v1 = values[i].x;
    auto v2 = values[i].y;
    auto v3 = values[i].z;
    auto v4 = values[i].w;
    values[i].x = v4;
    values[i].y = v3;
    values[i].z = v2;
    values[i].w = v1;
  }
}

template<bool TFlat, bool TSoA, typename T, typename std::enable_if<!TFlat && TSoA,bool>::type = 0>
__global__
void dkernel(SoA4<T> values, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x)
  {
    T v1 = values.x[i];
    T v2 = values.y[i];
    T v3 = values.z[i];
    T v4 = values.w[i];
    values.x[i] = v4;
    values.y[i] = v3;
    values.z[i] = v2;
    values.w[i] = v1;
  }
}


// ____________ init + check _________________________________________________


struct InitSoA {
  template<typename T>
  void operator()(T** dv_el, T** v_el, int n) {
    *v_el = new T[n];
    for(int i=0; i<n; ++i) {
      (*v_el)[i] = 1;
    }
    CHECK_CUDA( cudaMalloc(dv_el, n*sizeof(T)) );
    CHECK_CUDA( cudaMemcpy(*dv_el, *v_el, n*sizeof(T), cudaMemcpyDefault) );
  }
};
struct FinishSoA {
  template<typename T>
  void operator()(T** dv_el, T** v_el, int n) {
    CHECK_CUDA( cudaMemcpy(*v_el, *dv_el, n*sizeof(T), cudaMemcpyDefault) );
    for(int i=0; i<n; ++i)
      if((*v_el)[i]!=1) throw 1;
    CHECK_CUDA( cudaFree(*dv_el) );
    delete[] *v_el;
    *dv_el = nullptr;
    *v_el = nullptr;
  }
};

template<typename TFunctor, typename T>
void applyFuncSoA(SoA4<T>& dv, SoA4<T>& v, int n) {
  TFunctor()(&dv.x, &v.x, n);
  TFunctor()(&dv.y, &v.y, n);
  TFunctor()(&dv.z, &v.z, n);
  TFunctor()(&dv.w, &v.w, n);
}


template<bool TFlat, typename T, typename std::enable_if<TFlat==true,bool>::type = 0>
void init(vector<T>& v) {
  for(auto& val : v)
    val = 1;
}
template<bool TFlat, typename T, typename std::enable_if<TFlat==false,bool>::type = 0>
void init(vector<T>& v) {
  for(auto& val : v){
    val.x = 1;
    val.y = 1;
    val.z = 1;
    val.w = 1;
  }
}

template<bool TFlat, typename T, typename std::enable_if<TFlat==true,bool>::type = 0>
void check(const vector<T>& v)  {
  for(const auto& val : v){
    if(val != 1)
      throw 1;
  }
}
template<bool TFlat, typename T, typename std::enable_if<TFlat==false,bool>::type = 0>
void check(const vector<T>& v)  {
  for(const auto& val : v){
    if(val.x != 1 || val.y != 1 || val.z != 1 || val.w != 1)
      throw 1;
  }
}

// __________________ run ____________________________________________________
/**
 * @tparam T type (int, float)
 * @tparam RUNS number of kernel runs
 * @param n number of 4-dimensional vectors
 * @return average kernel time in milliseconds
 */
template<typename T, int TRUNS, bool TFlat, bool TSoA>
float run(int blocks, int threads, int n) {
  static_assert(TFlat==true || TSoA==false, "SoA<T> is a special case.");
  /// W = 4 if we are using int or float, so we need 4 * n ints / floats
  static constexpr int W = TFlat ? 4 : 1;
  /// type is int or float or a 4-dim. vector (int4, float4) depending onf TFlat parameter
  using Type = typename std::conditional<TFlat, T, typename std::conditional<std::is_same<T, int>::value, int4, float4>::type>::type;

  vector<Type> values_h(W*n);
  size_t bytes = W*n*sizeof(Type);
  Type* values_d;

  init<TFlat>(values_h);
  CHECK_CUDA( cudaMalloc(&values_d, bytes) );
  CHECK_CUDA( cudaMemcpy(values_d, values_h.data(), bytes, cudaMemcpyDefault) );
  CHECK_CUDA( cudaDeviceSynchronize() );
  CHECK_CUDA( cudaEventRecord(cstart) );

  for(int k=0; k<TRUNS; ++k)
    dkernel<TFlat, TSoA><<<blocks, threads>>>(values_d, n);

  CHECK_CUDA( cudaEventRecord(cstop) );
  CHECK_LAST("Kernel launch failed.");
  CHECK_CUDA( cudaMemcpy(values_h.data(), values_d, bytes, cudaMemcpyDefault) );
  CHECK_CUDA( cudaFree(values_d) );
  check<TFlat>(values_h);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, cstart, cstop);
  return milliseconds/TRUNS;
}

template<typename T, int TRUNS>
float runSoA(int blocks, int threads, int n) {
  using TSoA4 = SoA4<T>;
  TSoA4 values_h;
  TSoA4 values_d;
  applyFuncSoA<InitSoA>(values_d, values_h, n);
  CHECK_CUDA( cudaDeviceSynchronize() );
  CHECK_CUDA( cudaEventRecord(cstart) );
  for(int k=0; k<TRUNS; ++k)
    dkernel<false, true><<<blocks, threads>>>(values_d, n);
  CHECK_CUDA( cudaEventRecord(cstop) );
  CHECK_LAST("Kernel launch failed.");

  applyFuncSoA<FinishSoA>(values_d, values_h, n); // throws 1 if check failed
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, cstart, cstop);
  return milliseconds/TRUNS;
}


void print_result(const char* info, int n, int nruns, float ms) {
  std::cout << info << ", " << nruns << " runs, "<< n << " objects, " << ms << " ms [avg], " << n/ms/1000000*2*4*4 << " GB/s [avg]" << std::endl;
}

// __________________ main ___________________________________________________

int main(int argc, const char** argv)
{
  int n = 1024*1024;
  int bfac = 128;
  static constexpr int NRUNS = 100;

  if(argc>=2)
    n = atoi(argv[1]);
  if(argc>=3)
    bfac=atoi(argv[2]);
  if(n<1)
    throw std::runtime_error("n should be >1");

  CHECK_CUDA( cudaSetDevice(0) );

  CHECK_CUDA( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
  cudaDeviceProp prop;
  CHECK_CUDA( cudaGetDeviceProperties(&prop, 0) );
  std::cout << prop.name
            << " with peak BW = "
            << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6
            << std::endl;

  int numSMs;
  CHECK_CUDA( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0) );
  int blocks = bfac*numSMs;
  int threads = 256;
  if(n<=1024) {
    std::cout << ">> Using only 1 block on 1 SM since n<=1024"
              << std::endl;
    blocks = 1;
  }
  CHECK_CUDA(cudaEventCreate(&cstart));
  CHECK_CUDA(cudaEventCreate(&cstop));

  try {
    float ms;

    // warmup
    run<int, 1, false, false>(blocks, threads, n);

    ms = run<int, NRUNS, false, false>(blocks, threads, n);
    print_result("AoS [int]     ", n, NRUNS, ms);

    ms = run<int, NRUNS, true, false>(blocks, threads, n);
    print_result("flat-AoS [int]", n, NRUNS, ms);

    ms = run<int, NRUNS, true, true>(blocks, threads, n);
    print_result("flat-SoA [int]", n, NRUNS, ms);

    ms = runSoA<int, NRUNS>(blocks, threads, n);
    print_result("SoA [int]     ", n, NRUNS, ms);

    // warmup
/*    run<float, 1, false, false>(blocks, threads, n);

    ms = run<float, NRUNS, false, false>(blocks, threads, n);
    print_result("AoS [float]     ", n, NRUNS, ms);

    ms = run<float, NRUNS, true, false>(blocks, threads, n);
    print_result("flat-AoS [float]", n, NRUNS, ms);

    ms = run<float, NRUNS, true, true>(blocks, threads, n);
    print_result("flat-SoA [float]", n, NRUNS, ms);

    ms = runSoA<float, NRUNS>(blocks, threads, n);
    print_result("SoA [float]     ", n, NRUNS, ms);*/

  }catch(const std::runtime_error& e){
    cout << "Error: " << e.what() << endl;
  }
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
