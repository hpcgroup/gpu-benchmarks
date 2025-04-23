Before compiling do these:

### Perlmutter
```sh
module load PrgEnv-cray cudatoolkit craype-accel-nvidia80 nccl
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
```
### Frontier
Using ROCM 6.1.3
```sh
module reset
module load PrgEnv-cray craype-accel-amd-gfx90a cpe/23.05 amd/6.1.3
module load cray-mpich/8.1.30
module load rocm/6.1.3
module load libfabric/1.20.1
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
```

Using ROCM 5.7.1
```sh
module reset
module load PrgEnv-cray craype-accel-amd-gfx90a cpe/23.05 amd/5.7.1
module load cray-mpich/8.1.28
module load rocm/5.7.1
module load libfabric/1.20.1
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
```