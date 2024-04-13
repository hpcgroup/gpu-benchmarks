CC	= cc
INC	= -I${ROCM_PATH}/include
CFLAGS  = -std=c++11 -O3 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip
LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64 -lhipblas

all: gemm.x

gemm.x: gemm.cu 
	${CC} ${CFLAGS} ${INC} ${LDFLAGS} -o gemm.x gemm.cu

clean: 
	rm -f gemm.x 

