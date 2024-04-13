#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include "../fp16_conversion.h"

using namespace std;

#define FP16MM

const char* hipblasGetErrorString(hipblasStatus_t status)
{
    switch(status)
    {
        case HIPBLAS_STATUS_SUCCESS: return "HIPBLAS_STATUS_SUCCESS";
        case HIPBLAS_STATUS_NOT_INITIALIZED: return "HIPBLAS_STATUS_NOT_INITIALIZED";
        case HIPBLAS_STATUS_ALLOC_FAILED: return "HIPBLAS_STATUS_ALLOC_FAILED";
        case HIPBLAS_STATUS_INVALID_VALUE: return "HIPBLAS_STATUS_INVALID_VALUE"; 
        case HIPBLAS_STATUS_ARCH_MISMATCH: return "HIPBLAS_STATUS_ARCH_MISMATCH"; 
        case HIPBLAS_STATUS_MAPPING_ERROR: return "HIPBLAS_STATUS_MAPPING_ERROR";
        case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED"; 
        case HIPBLAS_STATUS_INTERNAL_ERROR: return "HIPBLAS_STATUS_INTERNAL_ERROR"; 
        case HIPBLAS_STATUS_NOT_SUPPORTED: return "HIPBLAS_STATUS_NOT_SUPPORTED";
        case HIPBLAS_STATUS_HANDLE_IS_NULLPTR: return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
        case HIPBLAS_STATUS_INVALID_ENUM: return "HIPBLAS_STATUS_INVALID_ENUM";
        case HIPBLAS_STATUS_UNKNOWN: return "HIPBLAS_STATUS_UNKNOWN";
    }
    return "unknown error";
}

// Convenience function for checking HIP runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
hipError_t checkHip(hipError_t result)
{
  if (result != hipSuccess) {
    fprintf(stderr, "HIP Runtime Error: %s\n", hipGetErrorString(result));
    assert(result == hipSuccess);
  }
  return result;
}

inline
hipblasStatus_t checkCublas(hipblasStatus_t result)
{
  if (result != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "HIP Runtime Error: %s\n", hipblasGetErrorString(result));
    assert(result == HIPBLAS_STATUS_SUCCESS);
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

#ifndef FP16MM
  cout << "\nhipblasSgemm test result:\n" << endl;
#else
  cout << "\nhipblasHgemm test result:\n" << endl;
#endif
  
  if(verbose) 
    cout << "running with" 
	 << " min_m_k_n: " << min_m_k_n
	 << " max_m_k_n: " << max_m_k_n
	 << " repeats: " << repeats
	 << endl;

  hipblasStatus_t stat;
  hipblasHandle_t handle;

  checkCublas(hipblasCreate(&handle));

  if(verbose) cout << "allocating device variables" << endl;
  
  // Allocate 3 arrays on CPU
  
  float *h_A = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
  float *h_B = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
  float *h_C = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));

  CPU_fill_rand(h_A, max_m_k_n, max_m_k_n);
  CPU_fill_rand(h_B, max_m_k_n, max_m_k_n);
  CPU_fill_rand(h_C, max_m_k_n, max_m_k_n);

#ifndef FP16MM
    // Allocate 3 arrays on GPU
    float *d_A, *d_B, *d_C;
    checkHip(hipMallocManaged(&d_A, max_m_k_n * max_m_k_n * sizeof(float)));
    checkHip(hipMallocManaged(&d_B, max_m_k_n * max_m_k_n * sizeof(float)));
    checkHip(hipMallocManaged(&d_C, max_m_k_n * max_m_k_n * sizeof(float)));
    
    checkHip(hipMemcpy(d_A,h_A,max_m_k_n * max_m_k_n * sizeof(float),hipMemcpyHostToDevice));
    checkHip(hipMemcpy(d_B,h_B,max_m_k_n * max_m_k_n * sizeof(float),hipMemcpyHostToDevice));
    checkHip(hipMemcpy(d_C,h_C,max_m_k_n * max_m_k_n * sizeof(float),hipMemcpyHostToDevice));
    
    int lda, ldb, ldc, m, n, k;
    const float alf = 1.0f;
    const float bet = 0.0f;
    const float *alpha = &alf;
    const float *beta = &bet;
  
#else
    
    uint16_t *d_A, *d_B, *d_C;
    checkHip(hipMallocManaged(&d_A, max_m_k_n * max_m_k_n * sizeof(uint16_t)));
    checkHip(hipMallocManaged(&d_B, max_m_k_n * max_m_k_n * sizeof(uint16_t)));
    checkHip(hipMallocManaged(&d_C, max_m_k_n * max_m_k_n * sizeof(uint16_t)));

    for (int i = 0; i < max_m_k_n * max_m_k_n; i++) {
        half temp_a = approx_float_to_half(h_A[i]);
        half temp_b = approx_float_to_half(h_B[i]);
        half temp_c = approx_float_to_half(h_C[i]);
        d_A[i] = *((uint16_t*) &temp_a);
        d_B[i] = *((uint16_t*) &temp_b);
        d_C[i] = *((uint16_t*) &temp_c);
    }

    int lda, ldb, ldc, m, n, k;
    half temp_alf = approx_float_to_half(1.0f);
    half temp_bet = approx_float_to_half(0.0f);
    const uint16_t alf = *((uint16_t*) &temp_alf);
    const uint16_t bet = *((uint16_t*) &temp_bet);
    const uint16_t *alpha = &alf;
    const uint16_t *beta = &bet;

#endif
  
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  for(int size = min_m_k_n; size <= max_m_k_n; size=size*2){
    double sum = 0.0;
    for(int rep = 0; rep < repeats; rep++){
      hipEventRecord(start, 0);
	  m=n=k=size;
	  lda = m;
	  ldb = k;
	  ldc = m;
#ifndef FP16MM
        stat = hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc); 
#else
	stat = hipblasHgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc); 
#endif
      hipEventRecord(stop,0);
      hipEventSynchronize(stop);
      if(stat != HIPBLAS_STATUS_SUCCESS){
	cerr << "hipblasSgemmBatched failed" << endl;
	exit(1);
      }
      assert(!hipGetLastError());
      
      float elapsed;
      hipEventElapsedTime(&elapsed, start, stop);
      elapsed /= 1000.0f;
      if (rep >= 25) {
          sum += elapsed;
      }
    }
#ifndef FP16MM	
  cout << "float32: size " 
#else
  cout << "float16: size " 
#endif
  << size << " average: " << sum/75 << " s "<< endl;

  }

  //Free GPU memory
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);

  // Free CPU memory
  free(h_A);
  free(h_B);
  free(h_C);
      
  return 0;
}

