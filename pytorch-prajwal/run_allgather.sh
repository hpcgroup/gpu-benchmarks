#!/bin/bash
#SBATCH -N 4
#SBATCH --account=csc569
#SBATCH --gpus=32
#SBATCH --time=00:20:00

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CRAY_MPICH_ROOTDIR}/gtl/lib"

## this enables the slingshot-11 plugin for RCCL (crucial for inter-node bw)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lustre/orion/scratch/prajwal/csc569/aws-ofi-rccl-rocm56/lib"
#export NCCL_DEBUG=INFO
export FI_CXI_ATS=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_GDR_LEVEL=3
## this improves cross node bandwidth for some cases
export NCCL_CROSS_NIC=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$((NNODES * 8))
export MASTER_ADDR=$(hostname)
#export MASTER_ADDR=
export MASTER_PORT=12345

#MIN_MSG_SIZE=$((1048576 * 2)) # 1048576 = 1024 * 1024
#MAX_MSG_SIZE=$((1048576 * 256))

# 32/64 GCD
MIN_MSG_SIZE=$((262144)) # 1048576 = 1024 * 1024
MAX_MSG_SIZE=$((1048576 * 64))


rm allgather/${GPUS}_gcd.out

SCRIPT="python -u allgather.py --num_gpus $GPUS --min_msg_size $MIN_MSG_SIZE --max_msg_size $MAX_MSG_SIZE --iterations 10"
#SCRIPT="python -u ground_allgather.py"
run_cmd="srun -N $NNODES -n $GPUS --cpu-bind=cores -c7 --gpus-per-task=1 --gpu-bind=closest ./set_env.sh $SCRIPT >> allgather/${GPUS}_gcd.out"

for i in {1..10}
do 
    echo $run_cmd
    eval $run_cmd
done
set +x
