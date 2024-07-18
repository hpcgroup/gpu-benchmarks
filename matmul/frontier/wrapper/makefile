CC      = cc
INC     = -I/opt/rocm-6.0.0/include
CFLAGS  = -std=c++17 -O3 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=/opt/rocm-6.0.0 --offload-arch=gfx90a -x hip -fPIC
LDFLAGS = -L/opt/rocm-6.0.0/lib -lamdhip64 -lrocblas -shared

all: gemm_wrapper.so

gemm_wrapper.so: gemm_wrapper.cu
	${CC} ${CFLAGS} ${INC} ${LDFLAGS} -o gemm_wrapper.so gemm_wrapper.cu

clean: 
	rm -f gemm_wrapper.so
