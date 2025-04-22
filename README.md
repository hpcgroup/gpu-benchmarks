Before compiling do these:

### Perlmutter
```sh
module load PrgEnv-cray cudatoolkit craype-accel-nvidia80 nccl
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
```
### Frontier
```sh
module load PrgEnv-cray amd-mixed/5.6.0 craype-accel-amd-gfx90a cray-mpich/8.1.26 cpe/23.05
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
```

