#define __HIP_PLATFORM_AMD__


#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include "../../fp16_conversion.h"
#include <torch/torch.h>
#include <torch/script.h>


using namespace std;


const char* rocblasGetErrorString(rocblas_status status)
{
    switch(status)
    {
        case rocblas_status_success: return "rocblas_status_success";
	case rocblas_status_invalid_handle: return "rocblas_status_invalid_handle";
	case rocblas_status_not_implemented: return "rocblas_status_not_implemented";
	case rocblas_status_invalid_pointer: return "rocblas_status_invalid_pointer";
	case rocblas_status_invalid_size: return "rocblas_status_invalid_size";
        case rocblas_status_memory_error: return "rocblas_status_memory_error";
	case rocblas_status_internal_error: return "rocblas_status_internal_error";
	case rocblas_status_perf_degraded: return "rocblas_status_perf_degraded";
	case rocblas_status_size_query_mismatch: return "rocblas_status_size_query_mismatch";
	case rocblas_status_size_increased: return "rocblas_status_size_increased";
	case rocblas_status_size_unchanged: return "rocblas_status_size_unchanged";
	case rocblas_status_invalid_value: return "rocblas_status_invalid_value";
	case rocblas_status_continue: return "rocblas_status_continue";
	case rocblas_status_check_numerics_fail: return "rocblas_status_check_numerics_fail";
        case rocblas_status_excluded_from_build: return "rocblas_status_excluded_from_build";
        case rocblas_status_arch_mismatch: return "rocblas_status_arch_mismatch";
    }
    return "unknown error";
}


inline
rocblas_status checkRocblas(rocblas_status result)
{
  if (result != rocblas_status_success) {
    fprintf(stderr, "ROCM Runtime Error: %s\n", rocblasGetErrorString(result));
    assert(result == rocblas_status_success);
  }
  return result;
}


void rocblas_gemm_ex_wrapper(int64_t transpose_A, int64_t transpose_B, int64_t m, int64_t n, int64_t k, const torch::Tensor& a, int64_t lda, const torch::Tensor& b, int64_t ldb, const torch::Tensor& c, int64_t ldc, torch::Tensor& d, int64_t ldd) {
        rocblas_handle handle;
        checkRocblas(rocblas_create_handle(&handle));

        rocblas_operation transA;
	if (transpose_A == 0) {
		transA = rocblas_operation_none;
	} else {
		transA = rocblas_operation_transpose;
	}

	rocblas_operation transB;
        if (transpose_B == 0) {
                transB = rocblas_operation_none;
	} else {
                transB = rocblas_operation_transpose;
        }


        half temp_alf = approx_float_to_half(1.0f);
        half temp_bet = approx_float_to_half(0.0f);
        const uint16_t alf = *((uint16_t*) &temp_alf);
        const uint16_t bet = *((uint16_t*) &temp_bet);
        const uint16_t *alpha = &alf;
        const uint16_t *beta = &bet;

	checkRocblas(
		rocblas_gemm_ex(
				handle, 
				transA, 
				transB, 
				m,
				n,
				k,
				alpha,
				a.data_ptr<at::Half>(),
				rocblas_datatype_bf16_r,
				lda,
				b.data_ptr<at::Half>(),
				rocblas_datatype_bf16_r,
				ldb,
				beta,
				c.data_ptr<at::Half>(),
				rocblas_datatype_bf16_r,
				ldc,
				d.data_ptr<at::Half>(),
				rocblas_datatype_bf16_r,
				ldd,
				rocblas_datatype_f32_r,
				rocblas_gemm_algo_standard,
				0,
				0)
        	);
}


TORCH_LIBRARY(gemm_wrapper, m) {
    m.def("rocblas_gemm_ex_wrapper", rocblas_gemm_ex_wrapper);
}
