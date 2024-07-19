#define __HIP_PLATFORM_AMD__

#include <torch/extension.h>
#include "torch/script.h"
#include "torch/torch.h"

#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

using namespace std;
rocblas_handle handle;

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

inline
hipError_t checkHip(hipError_t result)
{
  if (result != hipSuccess) {
    fprintf(stderr, "HIP Runtime Error: %s\n", hipGetErrorString(result));
    assert(result == hipSuccess);
  }
  return result;
}


void init_handle()
{
	checkRocblas(rocblas_create_handle(&handle));
}


void print_tensor(const torch::Tensor& x)
{
	std::cout << x << std::endl;
}

void matmul_AB(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C)
{
	// actually need to compute BT . AT in hipblas in NN format
	TORCH_CHECK(A.dim() == 2, "Matrix A should be 2 dimensional");	
	TORCH_CHECK(B.dim() == 2, "Matrix B should be 2 dimensional");
	TORCH_CHECK(C.dim() == 2, "Matrix C should be 2 dimensional");

	TORCH_CHECK(A.is_contiguous(), "Matrix A should be contiguous");	
	TORCH_CHECK(B.is_contiguous(), "Matrix B should be contiguous");
	TORCH_CHECK(C.is_contiguous(), "Matrix C should be contiguous");

	int lda, ldb, ldd, ldc, m, n, k;
	m = B.sizes()[1];
	k = B.sizes()[0];
	n = A.sizes()[0];
	ldb = B.sizes()[1];
	lda = A.sizes()[1];
	ldc = C.sizes()[1];	


	TORCH_CHECK(A.sizes()[1] == k, "Common dimension of A and B should be the same");
	TORCH_CHECK(C.sizes()[0] == n, "First dimension of C is incorrect");
	TORCH_CHECK(C.sizes()[1] == m, "Second dimension of C is incorrect");
	
	rocblas_status stat;
	float alpha = 1.0;
	float beta = 0.0;
	stat = rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none, 
			m, n, k, &alpha, 
			B.data_ptr<at::BFloat16>(), rocblas_datatype_bf16_r, ldb, 
			A.data_ptr<at::BFloat16>(), rocblas_datatype_bf16_r, lda, &beta, 
			C.data_ptr<at::BFloat16>(), rocblas_datatype_bf16_r, ldc, 
			C.data_ptr<at::BFloat16>(), rocblas_datatype_bf16_r, ldc, 
			rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
	checkRocblas(stat);

}


void matmul_ABT(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C)
{
	// actually need to compute B . AT in hipblas in TN format
	TORCH_CHECK(A.dim() == 2, "Matrix A should be 2 dimensional");	
	TORCH_CHECK(B.dim() == 2, "Matrix B should be 2 dimensional");
	TORCH_CHECK(C.dim() == 2, "Matrix C should be 2 dimensional");

	TORCH_CHECK(A.is_contiguous(), "Matrix A should be contiguous");	
	TORCH_CHECK(B.is_contiguous(), "Matrix B should be contiguous");
	TORCH_CHECK(C.is_contiguous(), "Matrix C should be contiguous");
	
	int lda, ldb, ldd, ldc, m, n, k;
	m = B.sizes()[0];
	k = B.sizes()[1];
	n = A.sizes()[0];
	
	ldb = B.sizes()[1];
	lda = A.sizes()[1];
	ldc = C.sizes()[1];	


	TORCH_CHECK(A.sizes()[1] == k, "Common dimension of A and B should be the same");
	TORCH_CHECK(C.sizes()[0] == n, "First dimension of C is incorrect");
	TORCH_CHECK(C.sizes()[1] == m, "Second dimension of C is incorrect");
	
	rocblas_status stat;
	float alpha = 1.0;
	float beta = 0.0;
	stat = rocblas_gemm_ex(handle, rocblas_operation_transpose, rocblas_operation_none, 
			m, n, k, &alpha, 
			B.data_ptr<at::BFloat16>(), rocblas_datatype_bf16_r, ldb, 
			A.data_ptr<at::BFloat16>(), rocblas_datatype_bf16_r, lda, &beta, 
			C.data_ptr<at::BFloat16>(), rocblas_datatype_bf16_r, ldc, 
			C.data_ptr<at::BFloat16>(), rocblas_datatype_bf16_r, ldc, 
			rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
	checkRocblas(stat);

}


void matmul_ATB(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C)
{
	// actually need to compute BT . A in hipblas in NT format
	TORCH_CHECK(A.dim() == 2, "Matrix A should be 2 dimensional");	
	TORCH_CHECK(B.dim() == 2, "Matrix B should be 2 dimensional");
	TORCH_CHECK(C.dim() == 2, "Matrix C should be 2 dimensional");

	TORCH_CHECK(A.is_contiguous(), "Matrix A should be contiguous");	
	TORCH_CHECK(B.is_contiguous(), "Matrix B should be contiguous");
	TORCH_CHECK(C.is_contiguous(), "Matrix C should be contiguous");
	
	int lda, ldb, ldd, ldc, m, n, k;
	m = B.sizes()[1];
	k = B.sizes()[0];
	n = A.sizes()[1];
	
	ldb = B.sizes()[1];
	lda = A.sizes()[1];
	ldc = C.sizes()[1];	


	TORCH_CHECK(A.sizes()[0] == k, "Common dimension of A and B should be the same");
	TORCH_CHECK(C.sizes()[0] == n, "First dimension of C is incorrect");
	TORCH_CHECK(C.sizes()[1] == m, "Second dimension of C is incorrect");
	
	rocblas_status stat;
	float alpha = 1.0;
	float beta = 0.0;
	stat = rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_transpose, 
			m, n, k, &alpha, 
			B.data_ptr<at::BFloat16>(), rocblas_datatype_bf16_r, ldb, 
			A.data_ptr<at::BFloat16>(), rocblas_datatype_bf16_r, lda, &beta, 
			C.data_ptr<at::BFloat16>(), rocblas_datatype_bf16_r, ldc, 
			C.data_ptr<at::BFloat16>(), rocblas_datatype_bf16_r, ldc, 
			rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
	checkRocblas(stat);

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("print_tensor", print_tensor);
  m.def("init_handle", init_handle);
  m.def("matmul_AB", matmul_AB);
  m.def("matmul_ABT", matmul_ABT);
  m.def("matmul_ATB", matmul_ATB);
}

