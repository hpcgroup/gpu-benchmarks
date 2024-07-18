#from mpi4py import MPI 
# somehow import MPI before torch gives weird HIP device not found errors
# so import torch first and then MPI
import torch
import torch.distributed as dist
from mpi4py import MPI
import time
import numpy as np
from argparse import ArgumentParser
import os


def torch_to_mpi(tensor: torch.Tensor):
    return [
        MPI.memory.fromaddress(
            tensor.data_ptr(), tensor.element_size() * tensor.nelement()
        ),
        MPI.FLOAT,
    ]

slurm_rank = int(os.getenv("SLURM_PROCID", 0))
torch.distributed.init_process_group(backend='nccl', rank=slurm_rank)
global_rank = torch.distributed.get_rank()
local_rank = global_rank % torch.cuda.device_count()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

if torch.distributed.get_rank() == 0:
    print("Starting MPI AllGather Benchmarking...")

NUM_TRIALS = 100

for H in [2048, 4096, 5120, 7168, 9216, 12288]:
    local_weight = torch.ones( H*4*H//world_size, device='cuda', dtype=torch.bfloat16)
    global_weight = torch.empty( H*4*H, device='cuda', dtype=torch.bfloat16)
    for _ in range(5):
        MPI.COMM_WORLD.Allgather(torch_to_mpi(local_weight), torch_to_mpi(global_weight))
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(NUM_TRIALS):
        MPI.COMM_WORLD.Allgather(torch_to_mpi(local_weight), torch_to_mpi(global_weight))
    
    torch.cuda.synchronize()
    end = time.time()

    if torch.distributed.get_rank() == 0:
        elapsed_time = end - start
        alg_bw = (world_size-1)/world_size * (H * 4 * H) * 2 * NUM_TRIALS / 1024 / 1024 / 1024 / elapsed_time
        print(f"GPUS = {world_size} | Output Message Size = {H*4*H*2/1024/1024:.3f} MB | Avg Time = {elapsed_time/NUM_TRIALS:.3f} s | alg_bw = {alg_bw:.3f} GBPS")

if torch.distributed.get_rank() == 0:
    print("======================")
    print("Starting xCCL AllGather Benchmarking...")

for H in [2048, 4096, 5120, 7168, 9216, 12288]:
    local_weight = torch.ones( H*4*H//world_size, device='cuda', dtype=torch.bfloat16)
    global_weight = torch.empty( H*4*H, device='cuda', dtype=torch.bfloat16)
    for _ in range(5):
        torch.distributed.all_gather_into_tensor(global_weight, local_weight)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(NUM_TRIALS):
        torch.distributed.all_gather_into_tensor(global_weight, local_weight)
    torch.cuda.synchronize()
    end = time.time()

    if torch.distributed.get_rank() == 0:
        elapsed_time = end - start
        alg_bw = (world_size-1)/world_size * (H * 4 * H) * 2 * NUM_TRIALS / 1024 / 1024 / 1024 / elapsed_time
        print(f"GPUS = {world_size} | Output Message Size = {H*4*H*2/1024/1024:.3f} MB | Avg Time = {elapsed_time/NUM_TRIALS:.3f} s | alg_bw = {alg_bw:.3f} GBPS")

torch.distributed.destroy_process_group()
