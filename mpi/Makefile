# Copyright 2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
# 
# SPDX-License-Identifier: MIT

CC = cc

# perlmutter flags
INC = -I/global/common/software/nersc9/nccl/2.19.4/include
CFLAGS = -std=c++11 -O2 -target-accel=nvidia80 --cuda-gpu-arch=sm_80 -DUSE_CUDA -DUSE_MPI
LDFLAGS = -L/global/common/software/nersc9/nccl/2.19.4/lib -lnccl

# frontier flags
# INC = -I${ROCM_PATH}/include
# CFLAGS = -std=c++11 -O2 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip -DUSE_ROCM -DUSE_MPI
# LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64 -lrccl

all: allgather.x allreduce.x reduce_scatter.x

allgather.x: ../allgather.cu
	${CC} ${CFLAGS} ${INC} ${LDFLAGS} -o all-gather/allgather.x ../allgather.cu

allreduce.x: ../allreduce.cu
	${CC} ${CFLAGS} ${INC} ${LDFLAGS} -o all-reduce/allreduce.x ../allreduce.cu

reduce_scatter.x: ../reduce_scatter.cu
	${CC} ${CFLAGS} ${INC} ${LDFLAGS} -o reduce-scatter/reduce_scatter.x ../reduce_scatter.cu

clean: 
	rm -f all-gather/allgather.x all-reduce/allreduce.x reduce-scatter/reduce_scatter.x
