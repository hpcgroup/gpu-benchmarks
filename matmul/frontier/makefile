CC	= cc
INC	= -I${ROCM_PATH}/include
CFLAGS  = -std=c++11 -O3 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip
LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64 -lrocblas

all: gemm-rectangular.x

gemm-rectangular.x: gemm-rectangular.cu 
	${CC} ${CFLAGS} ${INC} ${LDFLAGS} -o gemm-rectangular.x gemm-rectangular.cu

clean: 
	rm -f gemm-rectangular.x 

