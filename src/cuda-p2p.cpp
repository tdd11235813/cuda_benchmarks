
/**
 * Requires tasks=nodes * gpus_per_node (pinning by e.g. numactl)
 * Maybe mpirun is needed to run in order to set environment variables w.r.t. the MPI library.
 * If CUDA_VISIBLE_DEVICES is unset, call `mpirun bash -c 'CUDA_VISIBLE_DEVICES=0,1 ./mpi-cuda-aware`
 */

#include "cuda_helper.cuh"

// System includes
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>

void showDevices( )
{
  int totalDevices;
  CHECK_CUDA(cudaGetDeviceCount( &totalDevices ));
  std::stringstream ss;
  ss<< "\n"
    << "CUDA devices: " << totalDevices << "\n"
    << listCudaDevices().str();

  if(totalDevices>1){
    ss << "GPUPeerAccess (x=yes,o=no)\n";
    int accessifromj;
    for(int i=0; i<totalDevices; ++i)
      ss << i;
    ss << "\n";
    for(int i=0; i<totalDevices; ++i)
    {
      for(int j=0; j<totalDevices; ++j){
        if(i==j)
          ss << " ";
        else{
          CHECK_CUDA(cudaDeviceCanAccessPeer(&accessifromj, i, j));
          ss << (accessifromj?"x":"o");
        }
      }
      ss << "\n";
    }
  }
  std::cout << ss.str();
}


int main(int argc, char *argv[])
{
  showDevices();

  CHECK_CUDA(cudaDeviceReset());
  return 0;
}
