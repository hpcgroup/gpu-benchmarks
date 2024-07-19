#define __HIP_PLATFORM_AMD__

#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include "../fp16_conversion.h"

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
        // case rocblas_status_excluded_from_build: return "rocblas_status_excluded_from_build";
        // case rocblas_status_arch_mismatch: return "rocblas_status_arch_mismatch";
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
rocblas_status checkRocblas(rocblas_status result)
{
  if (result != rocblas_status_success) {
    fprintf(stderr, "ROCM Runtime Error: %s\n", rocblasGetErrorString(result));
    assert(result == rocblas_status_success);
  }
  return result;
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
void CPU_fill_rand(float *A, unsigned long long nr_rows_A, unsigned long long nr_cols_A) {
    int a=1;

    for (unsigned long long i = 0; i < nr_rows_A * nr_cols_A; i++){
        A[i] = (float)rand()/(float)(RAND_MAX/a);		
    }
}

int main(int argc, char ** argv) {

  int m = atoi(argv[1]);
  float k_mult = atof(argv[2]);
  float n_mult = atof(argv[3]);
  float h = atof(argv[4]);

  int k = k_mult * h;
  int n = n_mult * h;

  rocblas_status stat;
  rocblas_handle handle;

  checkRocblas(rocblas_create_handle(&handle));

  
  // Allocate 3 arrays on CPU

  float *h_A = (float *)malloc(m * k * sizeof(float));
  float *h_B = (float *)malloc(k * n * sizeof(float));
  float *h_C = (float *)malloc(m * n * sizeof(float));

  CPU_fill_rand(h_A, m, k);
  CPU_fill_rand(h_B, k, n);
  CPU_fill_rand(h_C, m, n);

    // Allocate 3 arrays on GPU
    uint16_t *d_A, *d_B, *d_C, *d_D;
    checkHip(hipMalloc(&d_A, m * k * sizeof(uint16_t)));
    checkHip(hipMalloc(&d_B, k * n * sizeof(uint16_t)));
    checkHip(hipMalloc(&d_C, m * n * sizeof(uint16_t)));
    
    // rocblas_gemm_ex requires D array too
    checkHip(hipMalloc(&d_D, m * n * sizeof(uint16_t)));

    for (unsigned long long i = 0; i < m * k; i++) {
        half temp_a = approx_float_to_half(h_A[i]);
        d_A[i] = *((uint16_t*) &temp_a);
    }

    for (unsigned long long i = 0; i < k * n; i++) {
        half temp_b = approx_float_to_half(h_B[i]);
        d_B[i] = *((uint16_t*) &temp_b);
    }

    for (unsigned long long i = 0; i < m * n; i++) {
        half temp_c = approx_float_to_half(h_C[i]);
        d_C[i] = *((uint16_t*) &temp_c);
    }


    int lda, ldb, ldc, ldd;
    half temp_alf = approx_float_to_half(1.0f);
    half temp_bet = approx_float_to_half(0.0f);
    const uint16_t alf = *((uint16_t*) &temp_alf);
    const uint16_t bet = *((uint16_t*) &temp_bet);
    const uint16_t *alpha = &alf;
    const uint16_t *beta = &bet;

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  int repeats = 15;

	      double sum = 0.0;
	      for(int rep = 0; rep < repeats; rep++) {
	          hipEventRecord(start, 0);

                  lda = m;
                  ldb = k;
                  ldc = m;
                  ldd = m;

                  stat = rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, alpha, d_A, rocblas_datatype_bf16_r, lda, d_B, rocblas_datatype_bf16_r, ldb, beta, d_C, rocblas_datatype_bf16_r, ldc, d_D, rocblas_datatype_bf16_r, ldd, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);

                  hipEventRecord(stop,0);
                  hipEventSynchronize(stop);
                  if(stat != rocblas_status_success){
                      fprintf(stderr, "RocBLAS Error: %s\n", rocblasGetErrorString(stat));
                      exit(1);
                  }
                  assert(!hipGetLastError());

                  float elapsed;
                  hipEventElapsedTime(&elapsed, start, stop);
                  elapsed /= 1000.0f;
                  if (rep >= 5) {
                      sum += elapsed;
                  }
              }

              double percent_of_peak = ((((((2 * ((double) k)) - 1) * ((double) m) * ((double) n)) / 1000000000000.0) / (sum / 10)) / 191.5) * 100;

	      cout << "m: " << m << " | k: " << k << " | n: " << n;  
	      cout << " | average: " << sum/75 << " s "<< " | Percent of Peak: " << percent_of_peak << endl;
 

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

