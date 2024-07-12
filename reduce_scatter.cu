/* \file reduce_scatter.cu
 * Copyright 2024 Parallel Software and Systems Group, University of Maryland.
 * See the top-level LICENSE file for details.
 * 
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdint.h>

#ifdef USE_CUDA
  #include <cuda_bf16.h>
  #define bfloat16 nv_bfloat16
#elif USE_ROCM
  #define __HIP_PLATFORM_AMD__
  #include <hip/hip_bfloat16.h>
  #include <hip/hip_runtime.h>
  #include <hip/hip_runtime_api.h>
  #define bfloat16 hip_bfloat16
#endif

#ifdef USE_NCCL
  #include "nccl.h"
#elif USE_RCCL
  #include <rccl/rccl.h> 
#endif

#define NUM_WARMUP_ITERATIONS		5

#define MPI_CHECK(cmd) do {                         \
  int64_t e = cmd;                                  \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%ld'\n",       \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDA_CHECK(cmd) do {                        \
  cudaError_t e = cmd;                              \
  if(e != cudaSuccess) {                            \
    printf("CUDA error  %s:%d: %s\n",               \
        __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define HIP_CHECK(cmd) do {                         \
  hipError_t e = cmd;                               \
  if(e != hipSuccess) {                             \
    printf("HIP error  %s:%d: %s\n",                \
        __FILE__, __LINE__, hipGetErrorString(e));  \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// NCCL_CHECK is used to validate RCCL functions as well
#define NCCL_CHECK(cmd) do {                        \
  ncclResult_t e = cmd;                             \
  if (e != ncclSuccess) {                           \
    printf("NCCL error %s:%d %s\n",                 \
        __FILE__, __LINE__, ncclGetErrorString(e)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void initializeData(bfloat16 *data, int64_t size) {
    for (int64_t i = 0; i < (size / sizeof(bfloat16)); ++i) {
        #ifdef USE_CUDA
        data[i] = __float2bfloat16((float)i);
        #elif USE_ROCM
        // ROCm doesn't have a float2bfloat16 method
        data[i] = (bfloat16) ((float) i);
        #endif
    }
}

void custom_bf16_sum(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
    bfloat16* in = (bfloat16*) invec;
    bfloat16* inout = (bfloat16*) inoutvec;
    for (int i = 0; i < *len; i++) {
        #ifdef USE_CUDA
        inout[i] = __hadd(in[i], inout[i]);
        #elif USE_ROCM
        inout[i] = in[i] + inout[i];
        #endif
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <num_gpus> <min_msg_size> <max_msg_size> <iterations>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int num_gpus = atoi(argv[1]);
    int64_t min_msg_size = strtoll(argv[2], NULL, 10);
    int64_t max_msg_size = strtoll(argv[3], NULL, 10);
    int iterations = atoi(argv[4]);

    if (num_gpus < 2 || min_msg_size <= 0 || max_msg_size <= 0 || min_msg_size > max_msg_size || iterations <= 0) {
        fprintf(stderr, "Invalid input parameters.\n");
        return EXIT_FAILURE;
    }

    int my_rank, num_pes;
    int num_gpus_per_node;
    int msg_count;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

    if (num_pes != num_gpus) {
        fprintf(stderr, "Number of processes must match number of GPUs.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Initialize GPU context
    #if USE_CUDA
    cudaGetDeviceCount(&num_gpus_per_node);
    cudaSetDevice((my_rank % num_gpus_per_node));
    #elif USE_ROCM
    hipGetDeviceCount(&num_gpus_per_node);
    hipSetDevice((my_rank % num_gpus_per_node));
    #endif

    int64_t local_data_size = max_msg_size; // Size of local data
    int64_t global_data_size = local_data_size; // Size of global data

    if (my_rank == 0) {
        fprintf(stdout, "Local data size: %ld\n", (local_data_size / 1024) / 1024);
        fprintf(stdout, "Global data size: %ld\n", (global_data_size / 1024) / 1024);
    }

    bfloat16 *local_data = (bfloat16*)malloc(local_data_size);
    bfloat16 *global_data = (bfloat16*)malloc(global_data_size);

    // Initialize local data
    initializeData(local_data, local_data_size);

    bfloat16 *d_local_data, *d_global_data;
    #ifdef USE_CUDA
    CUDA_CHECK(cudaMalloc(&d_local_data, local_data_size));
    CUDA_CHECK(cudaMalloc(&d_global_data, global_data_size));
    // Copy local data to GPU
    CUDA_CHECK(cudaMemcpy(d_local_data, local_data, local_data_size, cudaMemcpyHostToDevice));

    #elif USE_ROCM
    HIP_CHECK(hipMalloc(&d_local_data, local_data_size));
    HIP_CHECK(hipMalloc(&d_global_data, global_data_size));
    HIP_CHECK(hipMemcpy(d_local_data, local_data, local_data_size, hipMemcpyHostToDevice));
    #endif

    #ifdef USE_MPI
    // create 2-byte datatype (send raw, un-interpreted bytes)
    MPI_Datatype mpi_type_bfloat16;
    MPI_Type_contiguous(2, MPI_BYTE, &mpi_type_bfloat16);
    MPI_Type_commit(&mpi_type_bfloat16);

    // define custom reduce operation for nv_bfloat16 types
    MPI_Op CUSTOM_SUM;
    MPI_Op_create(&custom_bf16_sum, 1, &CUSTOM_SUM);

    #elif defined(USE_NCCL) || defined(USE_RCCL)
    ncclUniqueId nccl_comm_id;
    ncclComm_t nccl_comm;

    if (my_rank == 0) {
        /* Generates an Id to be used in ncclCommInitRank. */
        ncclGetUniqueId(&nccl_comm_id);
    }

    /* distribute nccl_comm_id to all ranks */
    MPI_CHECK(MPI_Bcast((void *)&nccl_comm_id, sizeof(nccl_comm_id), MPI_BYTE,
                        0, MPI_COMM_WORLD));

    /* Create a new NCCL/RCCL communicator */
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, num_pes, nccl_comm_id, my_rank));
    #endif

    // init recvcounts, which stores the portion of data to send to each process after calling reduce
    int *recvcounts = (int*) malloc(sizeof(int) * num_pes);
    int portion;

    // Perform MPI_Ireduce_scatter, NCCL reduce_scatter, or RCCL reduce_scatter 
    double total_time, start_time;
    MPI_Request request;
    MPI_Status status;

    // Print benchmark results
    if (my_rank == 0) {
        printf("Number of GPUs: %d\n", num_gpus);
        printf("Message size range: %ld - %ld\n", min_msg_size, max_msg_size);
        printf("Number of iterations: %d\n", iterations);
    }
    fflush(NULL);

    for (int64_t msg_size = min_msg_size; msg_size <= max_msg_size; msg_size *= 2) {
	msg_count = msg_size / sizeof(bfloat16);

    portion = msg_count / num_pes;
    for (int i = 0; i < num_pes; i++) 
        recvcounts[i] = portion;

	// warmup iterations
	for (int i = 0; i < NUM_WARMUP_ITERATIONS; ++i) {
            #ifdef USE_MPI
            MPI_CHECK(MPI_Ireduce_scatter(d_local_data, d_global_data, recvcounts, mpi_type_bfloat16,
                CUSTOM_SUM, MPI_COMM_WORLD, &request));

            MPI_CHECK(MPI_Wait(&request, &status));
            #elif defined(USE_NCCL) || defined(USE_RCCL)
            NCCL_CHECK(ncclReduceScatter((const void*)d_local_data, (void*)d_global_data, portion, ncclBfloat16, ncclSum, nccl_comm, NULL));
            #endif
            
            #ifdef USE_CUDA
            cudaDeviceSynchronize();
            #elif USE_ROCM
            hipDeviceSynchronize();
            #endif
        }

        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
	for (int i = 0; i < iterations; ++i) {
            #ifdef USE_MPI
            MPI_CHECK(MPI_Ireduce_scatter(d_local_data, d_global_data, recvcounts, mpi_type_bfloat16,
                CUSTOM_SUM, MPI_COMM_WORLD, &request));

            MPI_CHECK(MPI_Wait(&request, &status));
            #elif defined(USE_NCCL) || defined(USE_RCCL)
            NCCL_CHECK(ncclReduceScatter((const void*)d_local_data, (void*)d_global_data, portion, ncclBfloat16, ncclSum, nccl_comm, NULL));
            #endif
            
            #ifdef USE_CUDA
            cudaDeviceSynchronize();
            #elif USE_ROCM
            hipDeviceSynchronize();
            #endif
        }
        MPI_Barrier(MPI_COMM_WORLD);
        total_time = MPI_Wtime() - start_time;
	if (my_rank == 0)
	    printf("%ld %.6f seconds\n", msg_size, (total_time / iterations));
    }

    // Cleanup
    free(local_data);
    free(global_data);
    #ifdef USE_CUDA
    CUDA_CHECK(cudaFree(d_local_data));
    CUDA_CHECK(cudaFree(d_global_data));
    #elif USE_ROCM
    HIP_CHECK(hipFree(d_local_data));
    HIP_CHECK(hipFree(d_global_data));
    #endif

    #ifdef defined(USE_NCCL) || defined(USE_RCCL)
    ncclCommDestroy(nccl_comm);
    #endif

    MPI_Finalize();
    return EXIT_SUCCESS;
}
