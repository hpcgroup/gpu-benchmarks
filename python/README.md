# MPI and RCCL benchmarks using mpi4py and torch.distributed


## Instructions to build mpi4py on Frontier

```
#!/bin/bash
module load PrgEnv-cray
module load amd-mixed/6.0.0
module load cray-mpich/8.1.26
module load cpe/23.05
module load craype-accel-amd-gfx90a

export MPICH_GPU_SUPPORT_ENABLED=1
source activate ./frontier_conda_60/

INC="-I${ROCM_PATH}/include"
LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

MPICC="cc -shared" INC=$INC LDFLAGS=$LDFLAGS pip install --no-cache-dir --no-binary=mpi4py mpi4py


```


