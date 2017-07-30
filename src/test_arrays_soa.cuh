#include <iostream>
#include <vector>
#include <vector_types.h>
#include <type_traits>
using namespace std;


template<typename T>
struct SoA2 {
  using type = T;
  T* x;
  T* y;
};
template<typename T>
struct SoA3 {
  using type = T;
  T* x;
  T* y;
  T* z;
};
template<typename T>
struct SoA4 {
  using type = T;
  T* x;
  T* y;
  T* z;
  T* w;
};
struct InitSoA {
  template<typename T>
  void operator()(T** dv_el, T** v_el, int n) {
    *v_el = new T[n];
    for(int i=0; i<n; ++i) {
      (*v_el)[i] = 1;
    }
    cudaMalloc(dv_el, n*sizeof(T));
    cudaMemcpy(*dv_el, *v_el, n*sizeof(T), cudaMemcpyDefault);
  }
};
struct FinishSoA {
  template<typename T>
  void operator()(T** dv_el, T** v_el, int n) {
    cudaMemcpy(*v_el, *dv_el, n*sizeof(T), cudaMemcpyDefault);
    for(int i=0; i<n; ++i)
      if((*v_el)[i]!=1) throw 1;
    cudaFree(*dv_el);
    delete[] *v_el;
    *dv_el = nullptr;
    *v_el = nullptr;
  }
};

template<typename TFunctor, typename T>
void applyFuncSoA(SoA2<T>& dv, SoA2<T>& v, int n) {
  TFunctor()(&dv.x, &v.x, n);
  TFunctor()(&dv.y, &v.y, n);
}
template<typename TFunctor, typename T>
void applyFuncSoA(SoA3<T>& dv, SoA3<T>& v, int n) {
  TFunctor()(&dv.x, &v.x, n);
  TFunctor()(&dv.y, &v.y, n);
  TFunctor()(&dv.z, &v.z, n);
}
template<typename TFunctor, typename T>
void applyFuncSoA(SoA4<T>& dv, SoA4<T>& v, int n) {
  TFunctor()(&dv.x, &v.x, n);
  TFunctor()(&dv.y, &v.y, n);
  TFunctor()(&dv.z, &v.z, n);
  TFunctor()(&dv.w, &v.w, n);
}

template<typename T>
__device__ inline
void dperformSoA(SoA2<T>& values, int i) {
  T v1 = values.x[i];
  T v2 = values.y[i];
  values.x[i] = v2;
  values.y[i] = v1;
}
template<typename T>
__device__ inline
void dperformSoA(SoA3<T>& values, int i) {
  T v1 = values.x[i];
  T v2 = values.y[i];
  T v3 = values.z[i];
  values.x[i] = v3;
  values.y[i] = v2;
  values.z[i] = v1;
}
template<typename T>
__device__ inline
void dperformSoA(SoA4<T>& values, int i) {
  T v1 = values.x[i];
  T v2 = values.y[i];
  T v3 = values.z[i];
  T v4 = values.w[i];
  values.x[i] = v4;
  values.y[i] = v3;
  values.z[i] = v2;
  values.w[i] = v1;
}

template<typename TSOA>
__global__
void dkernelSoA(TSOA values, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x)
  {
    dperformSoA(values, i);
  }
}
/* Structure of Arrays */
template<typename TSOA, int RUNS>
void runSoA(int n) {
  using T=typename TSOA::type;
  TSOA values_h;
  TSOA values_d;

  int numSMs;
  cudaSetDevice(0);
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  dim3 blocks (32*numSMs);
  dim3 threads (256);
  if(n<=1024) {
    cout << ">> Using only 1 block on 1 SM since n<=1024" << endl;
    blocks.x = 1;
  }
  applyFuncSoA<InitSoA>(values_d, values_h, n);
  for(int k=0; k<RUNS; ++k)
    dkernelSoA<<<blocks, threads>>>(values_d, n);
  applyFuncSoA<FinishSoA>(values_d, values_h, n); // throws 1 if check failed
  cout << "Successful." << endl;
}
