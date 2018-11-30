
/**
 * Requires tasks=nodes * gpus_per_node (pinning by e.g. numactl)
 * Maybe mpirun is needed to run in order to set environment variables w.r.t. the MPI library.
 * If CUDA_VISIBLE_DEVICES is unset, call `mpirun bash -c 'CUDA_VISIBLE_DEVICES=0,1 ./mpi-cuda-aware`
 */

#include "cuda_helper.cuh"

// MPI include
#include <mpi.h>

// System includes
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>

#define TOSTR_(s)   #s
#define TOSTR(s)    TOSTR_(s)
#define CHECK_MPI(A) _mpi_check(A,#A,__FILE__,__LINE__)

/**
 * This is the environment variable which allows the reading of the local rank of the current MPI
 * process before the MPI environment gets initialized with MPI_Init(). This is necessary when running
 * the CUDA-aware MPI version in order to be able to
 * set the CUDA device for the MPI process before MPI environment initialization. If you are using MVAPICH2,
 * set this constant to "MV2_COMM_WORLD_LOCAL_RANK"; for Open MPI, use "OMPI_COMM_WORLD_LOCAL_RANK".
 *
 * This requires mpirun to launch this program.
 */
//#define ENV_LOCAL_RANK "MV2_COMM_WORLD_LOCAL_RANK"
//#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"

// Shut down MPI cleanly if something goes wrong
void abort_mpi(int err)
{
  std::cout << "Test FAILED\n";
  MPI_Abort(MPI_COMM_WORLD, err);
}

void _mpi_check(int c, const char* call, const char* file, int line)
{
  if(c!=MPI_SUCCESS){
    std::cerr << "MPI Error: "<< file<<":"<<line<<" "<<call<< " ("<<c<<")\n";
    abort_mpi(c);
  }
}

void SafeCheckMPIStatus(MPI_Status * status, int expectedElems)
{
  int recvElems;

  MPI_Get_count(status, MPI_FLOAT, &recvElems);

  if (recvElems != expectedElems)
  {
    fprintf(stderr, "Error: MPI transfer returned %d elements, but %d were expected. "
            "Terminating...\n", recvElems, expectedElems);
    abort_mpi(1);
  }
}

void showDevices( int rank )
{
  int namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(processor_name, &namelen);
  int totalDevices;
  CHECK_CUDA(cudaGetDeviceCount( &totalDevices ));
  std::stringstream ss;
  if(rank==0) {
    ss<< "\n"
      << "["<<rank<<":"<< processor_name<<"] "
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
}

void SetDeviceBeforeInit()
{
  char * localRankStr = NULL;
  int rank = 0, devCount = 0;

  // We extract the local rank initialization using an environment variable
  if ((localRankStr = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
    rank = atoi(localRankStr);
  } else if ((localRankStr = getenv("PMI_RANK")) != NULL) {
    rank = atoi(localRankStr);
  } else if ((localRankStr = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
    rank = atoi(localRankStr);
  } else {
    throw std::runtime_error("Local MPI rank environment variable required.");
  }

  CHECK_CUDA(cudaGetDeviceCount(&devCount));
  CHECK_CUDA(cudaSetDevice(rank % devCount));
}

void initialize(int * argc, char *** argv, int * rank, int * size)
{
  // Setting the device here will have an effect only for the CUDA-aware MPI version
  SetDeviceBeforeInit();

  CHECK_MPI(MPI_Init(argc, argv));
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, size));

  // direct = getenv("MPICH_RDMA_ENABLED_CUDA")==NULL?0:atoi(getenv ("MPICH_RDMA_ENABLED_CUDA"));
  // if(direct != 1){
  //   printf ("MPICH_RDMA_ENABLED_CUDA not enabled!\n");
  // }

  showDevices(*rank);
}

struct Bandwidth_Bidirectional {
  static constexpr const char*const unit = "GB/s";
  static constexpr const char*const bidirectional = "yes";
  double operator()(double repetitions, double size, double time) {
    return 2.0*repetitions*(1e-9*size) / time;
  }
};

struct Bandwidth {
  static constexpr const char*const unit = "GB/s";
  static constexpr const char*const bidirectional = "yes";
  double operator()(double repetitions, double size, double time) {
    return repetitions*(1e-9*size) / time;
  }
};

struct Latency_Bidirectional {
  static constexpr const char*const unit = "ms";
  static constexpr const char*const bidirectional = "yes";
  double operator()(double repetitions, double size, double time) {
    return time/repetitions/2.0*1e3; // s -> ms
  }
};

struct Latency {
  static constexpr const char*const unit = "ms";
  static constexpr const char*const bidirectional = "yes";
  double operator()(double repetitions, double size, double time) {
    return time/repetitions*1e3; // s -> ms
  }
};

struct Statistics {
  int comm_size;
  double size;
  double repetitions;
  double* times_all_nodes;

  Statistics()=delete;
  Statistics(int comm_size,
             size_t size,
             int repetitions,
             double* times_all_nodes)
    : comm_size(comm_size),
      size(static_cast<double>(size)),
      repetitions(static_cast<double>(repetitions)),
      times_all_nodes(times_all_nodes) {}

  template<typename T>
  std::stringstream compute() {
    std::stringstream ss;
    using std::setw;
    using std::setprecision;
    ss << "\n"
       << "Buffer size: " << size/1048576 << " MiB\n"
       << "Bidirectional: "<<T::bidirectional<<"\n"
       << "\n"
       << T::unit << "\n"
       << setw(2) << "|";
    for(int x=0; x<comm_size; ++x) {
      ss << setw(5) << x << " ";
    }
    ss << "\n";
    for(int y=0; y<comm_size; ++y) {
      ss << setw(1) << y << "|";
      for(int x=0; x<comm_size; ++x) {
        ss << setw(5) << setprecision(3);
        if(x==y)
          ss << "-";
        else
          ss << T()(repetitions,
                    size,
                    times_all_nodes[x+y*comm_size]);
        ss << " ";
      }
      ss << "\n";
    }
    ss << "\n";
    return ss;
  }
};

int main(int argc, char *argv[])
{
    // Get our MPI node number and node count
    int comm_size=0, comm_rank=0;
    initialize(&argc, &argv, &comm_rank, &comm_size);

    size_t data_size_per_node = 32*1048576;
    int elem_count_per_node = data_size_per_node/sizeof(float);
    int repetitions = 100;
    int warmups = 10;
    std::vector<double> times_node(comm_size);

    if(argc==2) {
      data_size_per_node = atoi(argv[1])*1048576; // $1 MB
    }

    // Allocate a buffer on each node
    // float *data_node_1 = new float[data_size_per_node];
    // float *data_node_2 = new float[data_size_per_node];
    size_t buffer_len = 128;
    float* host_buffer = new float[buffer_len];
    for(int k=0; k<buffer_len; ++k)
      host_buffer[k] = comm_rank;
    float* data_node_dev_send=nullptr;
    float* data_node_dev_recv=nullptr;
    CHECK_CUDA(cudaMalloc(&data_node_dev_send, data_size_per_node));
    CHECK_CUDA(cudaMalloc(&data_node_dev_recv, data_size_per_node));
    CHECK_CUDA(cudaMemcpy(data_node_dev_send, host_buffer, buffer_len*sizeof(float), cudaMemcpyHostToDevice));

    for(int rank_recv=0; rank_recv<comm_size; ++rank_recv) {
      if(rank_recv==comm_rank) {
        times_node[comm_rank] = 0.0;
        continue;
      }
      MPI_Status status;
      MPI_Request request;
      //std::cout << comm_rank << ": " << comm_rank << "<->" << rank_recv << "\n";

      double wtime;
      for(int r=-warmups; r<repetitions; ++r) {

        if(r==0)
          wtime = MPI_Wtime(); // start timer after warmup runs

        int tag_send = 1;
        int tag_recv = 2;
        if(comm_rank < rank_recv )
          MPI_Isend(data_node_dev_send, elem_count_per_node, MPI_FLOAT, rank_recv, tag_send, MPI_COMM_WORLD, &request);
        else
          MPI_Irecv(data_node_dev_recv, elem_count_per_node, MPI_FLOAT, rank_recv, tag_send, MPI_COMM_WORLD, &request);
        // faster
        MPI_Wait(&request, &status);
        SafeCheckMPIStatus(&status, elem_count_per_node);

        if(comm_rank < rank_recv )
          MPI_Irecv(data_node_dev_recv, elem_count_per_node, MPI_FLOAT, rank_recv, tag_recv, MPI_COMM_WORLD, &request);
        else
          MPI_Isend(data_node_dev_send, elem_count_per_node, MPI_FLOAT, rank_recv, tag_recv, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);
        SafeCheckMPIStatus(&status, elem_count_per_node);

        // bi-directional transfer of the elements - slower
        // MPI_Sendrecv(data_node_dev_send,
        //              elem_count_per_node,
        //              MPI_FLOAT,
        //              rank_recv,
        //              tag_send,
        //              data_node_dev_recv,
        //              elem_count_per_node,
        //              MPI_FLOAT,
        //              rank_recv,
        //              tag_recv,
        //              MPI_COMM_WORLD,
        //              &status);
        // SafeCheckMPIStatus(&status, elem_count_per_node);
        // std::cerr << "Sent\n";
        if(r==-warmups) {
          CHECK_CUDA(cudaMemcpy(host_buffer, data_node_dev_recv, buffer_len*sizeof(float), cudaMemcpyDeviceToHost));
          if(std::abs(host_buffer[12]-static_cast<float>(rank_recv))>1e-5)
            std::cerr << "Failed.\n";
            //std::cout << comm_rank << " ~ " << host_buffer[12] << "\n";
        }
      }
      times_node[rank_recv] = MPI_Wtime() - wtime;

//        = static_cast<double>(repetitions)
//        * (1e-9 * static_cast<double>(data_size_per_node) / wtime); // GB/s

//      std::cout << rank_recv <<": "<<times_node[rank_recv]<<" s\n";
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    double* times_all_nodes = nullptr;
    if( comm_rank==0 ) {
      times_all_nodes = new double[comm_size*comm_size];
      for(int i=0; i<comm_size; ++i)
        times_all_nodes[i] = times_node[i];
    }
    CHECK_MPI( MPI_Gather(times_node.data(), // sendbuffer
                          comm_size,  // sendcount
                          MPI_DOUBLE, // sendtype
                          times_all_nodes, // recvbuffer
                          comm_size,  // recvcount
                          MPI_DOUBLE, // recvtype
                          0,          // root rank
                          MPI_COMM_WORLD) );
    if( comm_rank==0 ) {
      Statistics statistics{
        comm_size,
        data_size_per_node,
        repetitions,
        times_all_nodes
      };
      std::cout << statistics.compute<Bandwidth_Bidirectional>().str();
      std::cout << statistics.compute<Latency_Bidirectional>().str();
    }

    // Cleanup
    if( comm_rank==0 ) {
      delete[] times_all_nodes;
    }
    delete[] host_buffer;
    CHECK_CUDA(cudaFree(data_node_dev_send));
    CHECK_CUDA(cudaFree(data_node_dev_recv));
    CHECK_MPI(MPI_Finalize());
    // after MPI_Finalize() which might frees internal CUDA buffers.
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
