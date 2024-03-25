# Copyright 2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
# 
# SPDX-License-Identifier: MIT

CC	= cc
INC	= -I/global/common/software/nersc9/nccl/2.19.4/include
CFLAGS	= -std=c++11 -O2 -target-accel=nvidia80 --cuda-gpu-arch=sm_80 -DUSE_CUDA -DUSE_NCCL
LDFLAGS	= -L/global/common/software/nersc9/nccl/2.19.4/lib -lnccl


all: allgather.x

allgather.x: allgather.cu 
	${CC} ${CFLAGS} ${INC} ${LDFLAGS} -o allgather.x allgather.cu

clean: 
	rm -f allgather.x 
