CC	= cc
CFLAGS	= -std=c++11 -O3 -target-accel=nvidia80 --cuda-gpu-arch=sm_80
LDFLAGS	= -L/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/math_libs/11.7/lib64 -lcublas -lcurand

all: gemm.x

gemm.x: gemm.cu 
	${CC} ${CFLAGS} ${LDFLAGS} -o gemm.x gemm.cu

clean: 
	rm -f gemm.x 

