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
struct SoA3 {
  using type = T;
  T* x;
  T* y;
  T* z;
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
    values[i] = v3;
    values[i+n] = v2;
    values[i+2*n] = v1;
  }
}

template<bool TFlat, bool TSoA, typename T, typename std::enable_if<TFlat && !TSoA,bool>::type = 0>
__global__
void dkernel(T* values, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x)
  {
    int s = 3*i;
    T v1 = values[s];
    T v2 = values[s+1];
    T v3 = values[s+2];
    values[s] = v3;
    values[s+1] = v2;
    values[s+2] = v1;
  }
}

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
    values[i].x = v3;
    values[i].y = v2;
    values[i].z = v1;
  }
}

template<bool TFlat, bool TSoA, typename T, typename std::enable_if<!TFlat && TSoA,bool>::type = 0>
__global__
void dkernel(SoA3<T> values, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x)
  {
    T v1 = values.x[i];
    T v2 = values.y[i];
    T v3 = values.z[i];
    values.x[i] = v3;
    values.y[i] = v2;
    values.z[i] = v1;
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
void applyFuncSoA(SoA3<T>& dv, SoA3<T>& v, int n) {
  TFunctor()(&dv.x, &v.x, n);
  TFunctor()(&dv.y, &v.y, n);
  TFunctor()(&dv.z, &v.z, n);
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
    if(val.x != 1 || val.y != 1 || val.z != 1 )
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

  static constexpr int W = TFlat ? 3 : 1;
  using Type = typename std::conditional<TFlat, T, typename std::conditional<std::is_same<T, int>::value, int3, float3>::type>::type;

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
  using TSoA3 = SoA3<T>;
  TSoA3 values_h;
  TSoA3 values_d;
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
  std::cout << info << ", " << nruns << " runs, "<< n << " objects, " << ms << " ms [avg], " << n/ms/1000000*2*4*3 << " GB/s [avg]" << std::endl;
}

// __________________ main ___________________________________________________

int main(int argc, const char** argv)
{
  int n = 1024*1024;
  static constexpr int NRUNS = 100;

  if(argc>=2)
    n = atoi(argv[1]);
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
  int blocks = 32*numSMs;
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
    CHECK_CUDA( cudaDeviceReset() );
  }
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
