#!/bin/bash
#SBATCH -p batch
#SBATCH -A CSC569
#SBATCH -t 10:00
#SBATCH -N 1
#SBATCH --output=/ccs/home/adityatomar/gpu-benchmarks/mpi/all-gather/frontier/benchmarks/8_gcd.txt
#SBATCH -C nvme

## calculating the number of nodes and GPUs
export NNODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=8 ## change as per your machine
export GPUS=$(( NNODES * GPUS_PER_NODE )) 

MIN_MSG_SIZE=$((1048576 * 2)) # 1048576 = 1024 * 1024
MAX_MSG_SIZE=$((1048576 * 256))
SCRIPT="/ccs/home/adityatomar/gpu-benchmarks/mpi/all-gather/allgather.x $GPUS $MIN_MSG_SIZE $MAX_MSG_SIZE 10"
run_cmd="srun -l -N $NNODES -n $GPUS -c7 --ntasks-per-node=8 --gpus-per-node=8 $SCRIPT" 

echo $run_cmd
eval $run_cmd
set +x
