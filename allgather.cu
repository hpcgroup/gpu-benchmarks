/* \file allgather.c
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
  #include <cuda_fp16.h>
#endif

#ifdef USE_NCCL
  #include "nccl.h"
#elif defined(USE_RCCL)
  #include "rccl.h"
#endif

#define NUM_GPU_DEVICES_PER_NODE	4
#define NUM_WARMUP_ITERATIONS		5

#define MPICHECK(cmd) do {                          \
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

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void initializeData(half *data, int size) {
    for (int i = 0; i < (size / sizeof(half)); ++i) {
        data[i] = __float2half((float)i);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <num_gpus> <min_msg_size> <max_msg_size> <iterations>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int num_gpus = atoi(argv[1]);
    int min_msg_size = atoi(argv[2]);
    int max_msg_size = atoi(argv[3]);
    int iterations = atoi(argv[4]);

    if (num_gpus < 2 || min_msg_size <= 0 || max_msg_size <= 0 || min_msg_size > max_msg_size || iterations <= 0) {
        fprintf(stderr, "Invalid input parameters.\n");
        return EXIT_FAILURE;
    }

    int my_rank, num_pes;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

    if (num_pes != num_gpus) {
        fprintf(stderr, "Number of processes must match number of GPUs.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Initialize GPU context
    cudaSetDevice((my_rank % NUM_GPU_DEVICES_PER_NODE));

    int local_data_size = max_msg_size; // Size of local data to be reduced
    int global_data_size = local_data_size * num_gpus; // Size of global data

    half *local_data = (half*)malloc(local_data_size);
    half *global_data = (half*)malloc(global_data_size);

    // Initialize local data
    initializeData(local_data, local_data_size);

    // Allocate memory on GPU
    half *d_local_data, *d_global_data;
    CUDA_CHECK(cudaMalloc(&d_local_data, local_data_size));
    CUDA_CHECK(cudaMalloc(&d_global_data, global_data_size));

    // Copy local data to GPU
    CUDA_CHECK(cudaMemcpy(d_local_data, local_data, local_data_size, cudaMemcpyHostToDevice));

    #ifdef USE_NCCL
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
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, num_pes, nccl_comm_id, rank));
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
        printf("Message size range: %d - %d\n", min_msg_size, max_msg_size);
        printf("Number of iterations: %d\n", iterations);
    }

    for (int msg_size = min_msg_size; msg_size <= max_msg_size; msg_size *= 2) {
        total_time = 0.0;

	// warmup iterations
	for (int i = 0; i < NUM_WARMUP_ITERATIONS; ++i) {
            #ifdef USE_MPI
	    MPICHECK(MPI_Iallgather(d_local_data, msg_size, MPI_HALF,
		d_global_data, msg_size, MPI_HALF, MPI_COMM_WORLD, &request));
                
            MPICHECK(MPI_Wait(&request, &status));
            #elif defined(USE_NCCL)
            NCCLCHECK(ncclAllGather((const void*)d_local_data, (void*)d_global_data, msg_size, ncclHalf, ncclSum, nccl_comm, NULL);
            #elif defined(USE_RCCL)
	    // TODO: fix later
            rcclAllReduce((const void*)d_local_data, (void*)d_global_data, global_data_size, rcclInt, rcclSum, comm, NULL);
            #endif
        }

        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
	for (int i = 0; i < iterations + 5; ++i) {
            #ifdef USE_MPI
            MPICHECK(MPI_Iallgather(d_local_data, msg_size, MPI_HALF,
                d_global_data, msg_size, MPI_HALF, MPI_COMM_WORLD, &request));

            MPICHECK(MPI_Wait(&request, &status));
            #elif defined(USE_NCCL)
            NCCLCHECK(ncclAllGather((const void*)d_local_data, (void*)d_global_data, msg_size, ncclHalf, ncclSum, nccl_comm, NULL);
            #elif defined(USE_RCCL)
            // TODO: fix later
            rcclAllReduce((const void*)d_local_data, (void*)d_global_data, global_data_size, rcclInt, rcclSum, comm, NULL);
            #endif
        }
        MPI_Barrier(MPI_COMM_WORLD);
        total_time = MPI_Wtime() - start_time;
        printf("%d %.6f seconds\n", msg_size, (total_time/iterations));
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

