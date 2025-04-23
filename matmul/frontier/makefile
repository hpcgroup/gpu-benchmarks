CC = hipcc
INC	= -I${ROCM_PATH}/include -I${MPICH_DIR}/include 
CFLAGS  = -std=c++11 -O3 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip
LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64 -lrocblas -L${MPICH_DIR}/lib -lmpi ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem \
			${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}

all: gemm_rocm-${CRAY_ROCM_VERSION}.x

gemm_rocm-${CRAY_ROCM_VERSION}.x: gemm.cu 
	${CC} ${CFLAGS} ${INC} ${LDFLAGS} -o gemm_rocm-${CRAY_ROCM_VERSION}.x gemm.cu

clean: 
	rm -f *.x 

