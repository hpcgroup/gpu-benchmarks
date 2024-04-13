#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../fp16_conversion.h"

using namespace std;

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
	case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

inline
cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
void CPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	int a=1;

    for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
          A[i] = (float)rand()/(float)(RAND_MAX/a);
	}
}

int main(int argc, char ** argv){


  int min_m_k_n = 1024;
  int max_m_k_n = 16384*2;
  int repeats = 100;
  int verbose = 1;

  if(verbose) 
    cout << "running with" 
	 << " min_m_k_n: " << min_m_k_n
	 << " max_m_k_n: " << max_m_k_n
	 << " repeats: " << repeats
	 << endl;

  cublasStatus_t stat;
  cublasHandle_t handle;

  checkCublas(cublasCreate(&handle));

  if(verbose) cout << "allocating device variables" << endl;
  
  // Allocate 3 arrays on CPU
  
  float *h_A = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
  float *h_B = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
  float *h_C = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));

  CPU_fill_rand(h_A, max_m_k_n, max_m_k_n);
  CPU_fill_rand(h_B, max_m_k_n, max_m_k_n);
  CPU_fill_rand(h_C, max_m_k_n, max_m_k_n);
  
  // Allocate 3 arrays on GPU
  nv_bfloat16 *d_A, *d_B, *d_C;
  checkCuda(cudaMallocManaged(&d_A, max_m_k_n * max_m_k_n * sizeof(nv_bfloat16)));
  checkCuda(cudaMallocManaged(&d_B, max_m_k_n * max_m_k_n * sizeof(nv_bfloat16)));
  checkCuda(cudaMallocManaged(&d_C, max_m_k_n * max_m_k_n * sizeof(nv_bfloat16)));    
  
  for (int i = 0; i < max_m_k_n * max_m_k_n; i++) {
    d_A[i] = __float2bfloat16(h_A[i]);
    d_B[i] = __float2bfloat16(h_B[i]);
    d_C[i] = __float2bfloat16(h_C[i]);
  }

  int lda, ldb, ldc, m, n, k;
  const nv_bfloat16 alf = __float2bfloat16(1.0f);
  const nv_bfloat16 bet = __float2bfloat16(0.0f);
  const nv_bfloat16 *alpha = &alf;
  const nv_bfloat16 *beta = &bet;
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for(int size = min_m_k_n; size <= max_m_k_n; size=size*2){
    double sum = 0.0;
    for(int rep = 0; rep < repeats; rep++){
      cudaEventRecord(start, 0);
	  m=n=k=size;
	  lda = m;
	  ldb = k;
	  ldc = m;
      
      stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, CUDA_R_16BF, lda, d_B, CUDA_R_16BF, ldb, beta, d_C, CUDA_R_16BF, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      if(stat != CUBLAS_STATUS_SUCCESS){
	  fprintf(stderr, "CuBLAS Error: %s\n", cublasGetErrorString(stat));
          exit(1);
      }
      assert(!cudaGetLastError());
      
      float elapsed;
      cudaEventElapsedTime(&elapsed, start, stop);
      elapsed /= 1000.0f;
      if (rep >= 25) {
          sum += elapsed;
      }
    }
  cout << "bfloat16: size " 
  << size << " average: " << sum/75 << " s "<< endl;

  }

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free CPU memory
  free(h_A);
  free(h_B);
  free(h_C);
      
  return 0;
}
