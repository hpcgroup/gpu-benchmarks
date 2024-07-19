CC	= cc
INC	= -I/opt/rocm-5.6.0/include
CFLAGS  = -std=c++17 -O3 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=/opt/rocm-5.6.0 --offload-arch=gfx90a -x hip
LDFLAGS = -L/opt/rocm-5.6.0/lib -lamdhip64 -lrocblas

all: gemm-rectangular.x

gemm-rectangular.x: gemm-rectangular.cu 
	${CC} ${CFLAGS} ${INC} ${LDFLAGS} -o gemm-rectangular.x gemm-rectangular.cu

clean: 
	rm -f gemm-rectangular.x 

