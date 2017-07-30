// from gpumembench

texture< int, 1, cudaReadModeElementType> texdataI1;
texture<int2, 1, cudaReadModeElementType> texdataI2;
texture<int4, 1, cudaReadModeElementType> texdataI4;

template<class T>
class dev_fun{
public:
	// Pointer displacement operation
	__device__ unsigned int operator()(T v1, unsigned int v2);
	// Compute operation (#1)
	__device__ T operator()(const T &v1, const T &v2);
	// Compute operation (#2)
	__device__ T comp_per_element(const T &v1, const T &v2);
	// Value initialization
	__device__ T init(int v);
	// Element loading
	__device__ T load(volatile const T* p, unsigned int offset);
	// Element storing
	__device__ void store(volatile T* p, unsigned int offset, const T &value);
	// Get first element
	__device__ int first_element(const T &v);
	// Reduce elements (XOR operation)
	__device__ int reduce(const T &v);
};


template<>
__device__ unsigned int dev_fun<int>::operator()(int v1, unsigned int v2){
	return v2+(unsigned int)v1 ;
}
template<>
__device__ int dev_fun<int>::operator()(const int &v1, const int &v2){
  return v1 + v2;
}
template<>
__device__ int dev_fun<int>::comp_per_element(const int &v1, const int &v2){
  return v1 - v2;
}
template<>
__device__ int dev_fun<int>::init(int v){
	return v;
}
template<>
__device__ int dev_fun<int>::load(volatile const int* p, unsigned int offset){
	int retval;
#ifdef TEX_LOADS
	retval = tex1Dfetch(texdataI1, offset);
#else
	p += offset;
	// Cache Operators for Memory Load Instructions
	// .ca Cache at all levels, likely to be accessed again.
	// .cg Cache at global level (cache in L2 and below, not L1).
	// .cs Cache streaming, likely to be accessed once.
	// .cv Cache as volatile (consider cached system memory lines stale, fetch again).
#ifdef L2_ONLY
	// Global level caching
	asm volatile ("ld.cg.u32 %0, [%1];" : "=r"(retval) : "l"(p));
#else
  // All cache levels utilized
  asm volatile ("ld.ca.u32 %0, [%1];" : "=r"(retval) : "l"(p));
//  asm volatile ("ld.global.nc.s32 %0, [%1];" : "=r"(retval) : "l"(p));
#endif
#endif
	return retval;
}
template<>
__device__ void dev_fun<int>::store(volatile int* p, unsigned int offset, const int &value){
	p += offset;
	// Cache Operators for Memory Store Instructions
	// .wb Cache write-back all coherent levels.
	// .cg Cache at global level (cache in L2 and below, not L1).
	// .cs Cache streaming, likely to be accessed once.
	// .wt Cache write-through (to system memory).

	// Streaming store
	asm volatile ("st.cs.global.u32 [%0], %1;" :: "l"(p), "r"(value));
}
template<>
__device__ int dev_fun<int>::first_element(const int &v){
	return v;
}
template<>
__device__ int dev_fun<int>::reduce(const int &v){
	return v;
}


template<>
__device__ unsigned int dev_fun<int2>::operator()(int2 v1, unsigned int v2){
	return v2+(unsigned int)(v1.x+v1.y) ;
}
template<>
__device__ int2 dev_fun<int2>::operator()(const int2 &v1, const int2 &v2){
	return make_int2(v1.x + v2.x, v1.y + v2.y);
}
template<>
__device__ int2 dev_fun<int2>::comp_per_element(const int2 &v1, const int2 &v2){
	return make_int2(v1.x - v2.x, v1.y - v2.y);
}
template<>
__device__ int2 dev_fun<int2>::init(int v){
	return make_int2(v, v);
}
template<>
__device__ int2 dev_fun<int2>::load(volatile const int2* p, unsigned int offset){
	union{
		unsigned long long ll;
		int2 i2;
	} retval;
#ifdef TEX_LOADS
	retval.i2 = tex1Dfetch(texdataI2, offset);
#else
	p += offset;
#ifdef L2_ONLY
	// Global level caching
	asm volatile ("ld.cg.u64 %0, [%1];" : "=l"(retval.ll) : "l"(p));
#else
	// All cache levels utilized
	asm volatile ("ld.ca.u64 %0, [%1];" : "=l"(retval.ll) : "l"(p));
#endif
#endif
	return retval.i2;
}
template<>
__device__ void dev_fun<int2>::store(volatile int2* p, unsigned int offset, const int2 &value){
	union{
		unsigned long long ll;
		int2 i2;
	} retval;
	retval.i2 = value;
	p += offset;
	// Streaming store
	asm volatile ("st.cs.global.u64 [%0], %1;" :: "l"(p), "l"(retval.ll));
}
template<>
__device__ int dev_fun<int2>::first_element(const int2 &v){
	return v.x;
}
template<>
__device__ int dev_fun<int2>::reduce(const int2 &v){
	return v.x ^ v.y;
}


template<>
__device__ unsigned int dev_fun<int4>::operator()(int4 v1, unsigned int v2){
	return v2+(unsigned int)(v1.x+v1.y+v1.z+v1.w) ;
}
template<>
__device__ int4 dev_fun<int4>::operator()(const int4 &v1, const int4 &v2){
	return make_int4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}
template<>
__device__ int4 dev_fun<int4>::comp_per_element(const int4 &v1, const int4 &v2){
	return make_int4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
}
template<>
__device__ int4 dev_fun<int4>::init(int v){
	return make_int4(v, v, v, v);
}
template<>
__device__ int4 dev_fun<int4>::load(volatile const int4* p, unsigned int offset){
	int4 retval;
#ifdef TEX_LOADS
	retval = tex1Dfetch(texdataI4, offset);
#else
	p += offset;
#ifdef L2_ONLY
	// Global level caching
	asm volatile ("ld.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(retval.x), "=r"(retval.y), "=r"(retval.z), "=r"(retval.w) : "l"(p));
#else
	// All cache levels utilized
	asm volatile ("ld.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(retval.x), "=r"(retval.y), "=r"(retval.z), "=r"(retval.w) : "l"(p));
#endif
#endif
	return retval;
}
template<>
__device__ void dev_fun<int4>::store(volatile int4* p, unsigned int offset, const int4 &value){
	p += offset;
	// Streaming store
	asm volatile ("st.cs.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) );
}
template<>
__device__ int dev_fun<int4>::first_element(const int4 &v){
	return v.x;
}
template<>
__device__ int dev_fun<int4>::reduce(const int4 &v){
	return v.x ^ v.y ^ v.z ^ v.w;
}
