#!/bin/bash

#SBATCH -A m4641_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 10:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export CUDA_DEVICE_MAX_CONNECTIONS=1
NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
export WORLD_SIZE=$GPUS
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1

MIN_MSG_SIZE=$((1048576 * 32)) # 1048576 = 1024 * 1024
MAX_MSG_SIZE=$((1048576 * 2048))

SCRIPT="$SCRATCH/gpu-benchmarks/mpi/reduce-scatter/reduce_scatter.x $GPUS $MIN_MSG_SIZE $MAX_MSG_SIZE 10"
run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 $SCRIPT >& $SCRATCH/gpu-benchmarks/mpi/reduce-scatter/perlmutter/benchmarks/16_gpu.txt"

echo $run_cmd
eval $run_cmd
set +x
