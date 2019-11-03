#!/bin/bash

WORKING_DIR=$HOME/cuda-workspace
RESULTS_DIR=$HOME/results
TEST_FFT=OFF
TEST_DCGM=OFF

# davide
# module load gnu
# module load cuda
# module load cmake
# module load openmpi
# module load boost
# module load fftw
# export CMAKE_PREFIX_PATH=$BOOST_HOME:$FFTW_HOME

# hemera
if [ "$(hostname)" == *"hemera" ]; then
    module load cmake
    module load cuda
    module load gcc # required otherwise below will not load
    module load openmpi/2.1.2-cuda100
    module load boost/1.67.0
fi

# START

lscpu > $RESULTS_DIR/results-lscpu.txt
lspci > $RESULTS_DIR/results-lspci.txt
nvidia-smi > $RESULTS_DIR/results-nsmi.txt
numactl -s > $RESULTS_DIR/results-cpubind.txt
numactl -H > $RESULTS_DIR/results-cpus.txt

cd ${WORKING_DIR}
[ -d "read-nvml-clocks-pci" ] || git clone --recursive https://github.com/tdd11235813/read-nvml-clocks-pci.git read-nvml-clocks-pci
cd read-nvml-clocks-pci/
make
./read-nvml-clocks-pci > $RESULTS_DIR/results-clocks-pci.txt


cd ${WORKING_DIR}
[ -d "cuda_benchmarks" ] || git clone --recursive https://github.com/tdd11235813/cuda_benchmarks cuda_benchmarks
cd cuda_benchmarks/
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=Release -DTEST_COOP_GROUPS=OFF ..
make -j

nvidia-smi topo -m | sed -r "s/\x1b\[[0-9;]*m?//g" > $RESULTS_DIR/results-topo.txt
./cuda-p2p > $RESULTS_DIR/results-p2p.txt
./cuda-sdk-p2p p > $RESULTS_DIR/results-sdk-p2p.txt
./cuda-sdk-p2p s > $RESULTS_DIR/results-sdk-p2p-sm.txt

for dev in 0 1 2 3; do for i in 1 2 4 8; do ./async_pcie $((i*16777216)) $dev >> $RESULTS_DIR/results-memcpy-$dev.txt; done; done

cd ../..

if [ "$TEST_FFT" == "ON" ]; then
    [ -d "gearshifft" ] || git clone --recursive https://github.com/tdd11235813/gearshifft ${WORKING_DIR}/gearshifft
    cd gearshifft/
    mkdir release
    cd release
    cmake -DCMAKE_BUILD_TYPE=Release -DGEARSHIFFT_FLOAT16_SUPPORT=1 -DFFTW_USE_STATIC_LIBS=ON -DBoost_USE_STATIC_LIBS=OFF ..
    make -j
    ./gearshifft/gearshifft_cufft -l
    ctest
    ./gearshifft/gearshifft_cufft -e 524288 -v
fi


if [ "$TEST_DCGM" == "ON" ]; then
    cd ${WORKING_DIR}
    mkdir dcgm
    # get it online: https://developer.nvidia.com/compute/DCGM/secure/1.46/RPMS/ppc64le/datacenter-gpu-manager-1.4.6-1.ppc64le
    # # tested on: davide
    rpm2cpio ./datacenter-gpu-manager-1.4.6-1.ppc64le.rpm | cpio -idmv
    cd usr/bin/
    # fix wrong-placed symbolic link
    rm nvvs
    ln -s ../share/nvidia-validation-suite/nvvs nvvs
    # run long tests (15-30min)
    ./nvvs -g -l
fi

# if needed, create config file nvvs.conf, to run medium tests
#
# $ cat nvvs.conf
# %YAML 1.2
# ---
# gpus:
# - gpuset: all_P100c
# properties:
# name: Tesla P100-SXM2-16GB
# tests:
#    - name: Medium
#
# ./nvvs -c nvvs.conf
