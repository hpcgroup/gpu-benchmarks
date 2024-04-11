Before compiling do these:

### Perlmutter
```sh
module load PrgEnv-cray cudatoolkit craype-accel-nvidia80
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
```
### Frontier
```sh
module load PrgEnv-cray amd-mixed craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1
```
