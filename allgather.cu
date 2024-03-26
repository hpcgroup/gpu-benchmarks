/* \file allgather.cu
 * Copyright 2024 Parallel Software and Systems Group, University of Maryland.
 * See the top-level LICENSE file for details.
 * 
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#ifdef USE_CUDA
  #include <cuda_runtime.h>
  #include <cuda_bf16.h>
#endif

#ifdef USE_NCCL
  #include "nccl.h"
#elif defined(USE_RCCL)
  #include "rccl.h"
#endif

#define NUM_WARMUP_ITERATIONS		5

#define MPI_CHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
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

#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t e = cmd;                             \
  if (e != ncclSuccess) {                           \
    printf("NCCL error %s:%d %s\n",                 \
        __FILE__, __LINE__, ncclGetErrorString(e)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void initializeData(nv_bfloat16 *data, int size) {
    for (int i = 0; i < (size / sizeof(nv_bfloat16)); ++i) {
        data[i] = __float2bfloat16((float)i);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <num_gpus> <min_msg_size> <max_msg_size> \n", argv[0]);
        return EXIT_FAILURE;
    }

    int num_gpus = atoi(argv[1]);
    int min_msg_size = atoi(argv[2]);
    int max_msg_size = atoi(argv[3]);

    if (num_gpus < 2 || min_msg_size <= 0 || max_msg_size <= 0 || min_msg_size > max_msg_size) {
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
    cudaGetDeviceCount(&num_gpus_per_node);
    cudaSetDevice((my_rank % num_gpus_per_node));

    int local_data_size = max_msg_size; // Size of local data
    int global_data_size = local_data_size * num_gpus; // Size of global data

    nv_bfloat16 *local_data = (nv_bfloat16*)malloc(local_data_size);
    nv_bfloat16 *global_data = (nv_bfloat16*)malloc(global_data_size);

    // Initialize local data
    initializeData(local_data, local_data_size);

    // Allocate memory on GPU
    nv_bfloat16 *d_local_data, *d_global_data;
    CUDA_CHECK(cudaMalloc(&d_local_data, local_data_size));
    CUDA_CHECK(cudaMalloc(&d_global_data, global_data_size));

    // Copy local data to GPU
    CUDA_CHECK(cudaMemcpy(d_local_data, local_data, local_data_size, cudaMemcpyHostToDevice));

    #ifdef USE_MPI
    // create 2-byte datatype (send raw, un-interpreted bytes)
    MPI_Datatype mpi_type_bfloat16;
    MPI_Type_contiguous(2, MPI_BYTE, &mpi_type_bfloat16);
    MPI_Type_commit(&mpi_type_bfloat16);

    #elif USE_NCCL
    ncclUniqueId nccl_comm_id;
    ncclComm_t nccl_comm;

    if (my_rank == 0) {
        /* Generates an Id to be used in ncclCommInitRank. */
        ncclGetUniqueId(&nccl_comm_id);
    }

    /* distribute nccl_comm_id to all ranks */
    MPI_CHECK(MPI_Bcast((void *)&nccl_comm_id, sizeof(nccl_comm_id), MPI_BYTE,
                        0, MPI_COMM_WORLD));
    /* Create a new NCCL communicator */
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, num_pes, nccl_comm_id, my_rank));

    #elif defined(USE_RCCL)
    // TODO: fix later
    rcclComm_t rccl_comm;
    rcclCommInitRank(&comm, num_gpus, 0, rccl_root);
    #endif

    // Perform MPI_Iallgather, NCCL allgather, or RCCL allgather
    double total_time, start_time;
    MPI_Request request;
    MPI_Status status;

    // Print benchmark results
    if (my_rank == 0) {
        printf("Number of GPUs: %d\n", num_gpus);
        printf("Message size range: %d - %d MB \n", min_msg_size / 1000000, max_msg_size / 1000000);
    }
    fflush(NULL);

    for (int msg_size = min_msg_size; msg_size <= max_msg_size; msg_size *= 2) {
	msg_count = msg_size / sizeof(nv_bfloat16);

	// warmup iterations
	for (int i = 0; i < NUM_WARMUP_ITERATIONS; ++i) {
            #ifdef USE_MPI
	    MPI_CHECK(MPI_Iallgather(d_local_data, msg_count, mpi_type_bfloat16,
		d_global_data, msg_count, mpi_type_bfloat16, MPI_COMM_WORLD, &request));
                
            MPI_CHECK(MPI_Wait(&request, &status));
            #elif defined(USE_NCCL)
            NCCL_CHECK(ncclAllGather((const void*)d_local_data, (void*)d_global_data, msg_count, ncclBfloat16, nccl_comm, NULL));
	    cudaDeviceSynchronize();
            #elif defined(USE_RCCL)
	    // TODO: fix later
            rcclAllReduce((const void*)d_local_data, (void*)d_global_data, global_data_size, rcclInt, rcclSum, comm, NULL);
            #endif
        }
	int iterations;
	if (msg_size <= 8000000) {
	    iterations = 25;
	}
	else {
	    iterations = 10;
	}
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();

	for (int i = 0; i < iterations; ++i) {
            #ifdef USE_MPI
            MPI_CHECK(MPI_Iallgather(d_local_data, msg_count, mpi_type_bfloat16,
                d_global_data, msg_count, mpi_type_bfloat16, MPI_COMM_WORLD, &request));

            MPI_CHECK(MPI_Wait(&request, &status));
	    
            #elif defined(USE_NCCL)
            NCCL_CHECK(ncclAllGather((const void*)d_local_data, (void*)d_global_data, msg_count, ncclBfloat16, nccl_comm, NULL));
	    cudaDeviceSynchronize();
            #elif defined(USE_RCCL)
            // TODO: fix later
            rcclAllReduce((const void*)d_local_data, (void*)d_global_data, global_data_size, rcclInt, rcclSum, comm, NULL);
            #endif
        }
        MPI_Barrier(MPI_COMM_WORLD);
        total_time = MPI_Wtime() - start_time;
	if (my_rank == 0)
	    printf("%d %.6f seconds\n", msg_size / 1000000, (total_time / iterations));
    }

    // Cleanup
    free(local_data);
    free(global_data);
    CUDA_CHECK(cudaFree(d_local_data));
    CUDA_CHECK(cudaFree(d_global_data));

    #ifdef USE_NCCL
    ncclCommDestroy(nccl_comm);
    #elif defined(USE_RCCL)
    rcclCommDestroy(rccl_comm);
    #endif

    MPI_Finalize();
    return EXIT_SUCCESS;
}
