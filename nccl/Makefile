# Copyright 2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
# 
# SPDX-License-Identifier: MIT

CC = cc

# perlmutter flags
INC = -I/global/common/software/nersc9/nccl/2.19.4/include
CFLAGS = -std=c++11 -O2 -target-accel=nvidia80 --cuda-gpu-arch=sm_80 -DUSE_CUDA -DUSE_NCCL
LDFLAGS = -L/global/common/software/nersc9/nccl/2.19.4/lib -lnccl

all: allgather.x allreduce.x reduce_scatter.x

allgather.x: ../allgather.cu
	${CC} ${CFLAGS} ${INC} ${LDFLAGS} -o all-gather/allgather.x ../allgather.cu

allreduce.x: ../allreduce.cu
	${CC} ${CFLAGS} ${INC} ${LDFLAGS} -o all-reduce/allreduce.x ../allreduce.cu

reduce_scatter.x: ../reduce_scatter.cu
	${CC} ${CFLAGS} ${INC} ${LDFLAGS} -o reduce-scatter/reduce_scatter.x ../reduce_scatter.cu

clean: 
	rm -f all-gather/allgather.x all-reduce/allreduce.x reduce-scatter/reduce_scatter.x
