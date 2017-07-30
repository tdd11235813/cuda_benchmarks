cmake_minimum_required (VERSION 2.8)
find_package(CUDA REQUIRED)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CUDA_VERSION_STRING VERSION_EQUAL 8.0 OR CUDA_VERSION_STRING VERSION_GREATER 8.0)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 6.0)
      message(FATAL_ERROR "CUDA 8.0 does not support gcc6+. Please choose an older gcc version like:\n
cmake -DCMAKE_C_COMPILER=`which gcc-5` -DCMAKE_CXX_COMPILER=`which g++-5` ..")
    endif()
  elseif(CUDA_VERSION_STRING VERSION_EQUAL 7.5 OR CUDA_VERSION_STRING VERSION_GREATER 7.5)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
      message(FATAL_ERROR "CUDA 7.5 does not support gcc5+. Please choose an older gcc version like:\n
cmake -DCMAKE_C_COMPILER=`which gcc-4` -DCMAKE_CXX_COMPILER=`which g++-4` ..")
    endif()
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "gcc5.0 or newer required.")
  endif()
endif()

set( CUDA_SEPARABLE_COMPILATION ON)
#list(APPEND CUDA_NVCC_FLAGS -std=c++11 -lineinfo --relocatable-device-code true)
list(APPEND CUDA_NVCC_FLAGS -std=c++11 -lineinfo)
#list(APPEND CUDA_NVCC_FLAGS "-g -G -lineinfo")
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_37,code=sm_37)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=compute_70)


#set(SOURCES test_arrays.cu)
# cuda_compile_ptx(
#   cuda_ptx_files
#   ${SOURCES}
#   OPTIONS --source-in-ptx
# )
# add_custom_target(ptx ALL
#   DEPENDS ${cuda_ptx_files}
#   SOURCES ${SOURCES}
# )

cuda_add_executable(coalescing coalescing.cu)
cuda_add_executable(arrays test_arrays.cu)
cuda_add_executable(arrays_3 test_arrays_3.cu)
cuda_add_executable(arrays_3_in_4 test_arrays_3_in_4.cu)
cuda_add_executable(arrays_zcopy test_zcopy.cu)
cuda_add_executable(ilp ilp.cu)
cuda_add_executable(pagesize test_pagesize.cu)
cuda_add_executable(pagesize_vert test_pagesize_vert.cu)
cuda_add_executable(async_pcie test_async_pcie.cu)
cuda_add_executable(caches test_caches.cu)

if(NOT CUDA_VERSION VERSION_LESS "9.0") # coop group examples require CC 6.0+
  cuda_add_executable(cuda9_groups cuda9_groups.cu)
  cuda_add_executable(cuda9_groups_memcpy cuda9_groups_memcpy.cu)
  cuda_add_executable(cuda9_without_groups cuda9_without_groups.cu)

  if(NOT CUDA_cudadevrt_LIBRARY)
    cuda_find_library_local_first(CUDA_cudadevrt_LIBRARY cudadevrt "\"cudadevrt\" library")
    mark_as_advanced(CUDA_cudadevrt_LIBRARY)
  endif()
  target_link_libraries(cuda9_groups ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY})
  target_link_libraries(cuda9_groups_memcpy ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY})
endif()
