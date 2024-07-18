import torch
from mpi4py import MPI
import os
import torch.distributed as dist

def init_dist():
    if not torch.distributed.is_initialized():
        init_method = "tcp://"
        master_ip = os.getenv("MASTER_ADDR", "localhost")
        master_port = os.getenv("MASTER_PORT", "6000")
        init_method += master_ip + ":" + master_port
        dist.init_process_group(
            backend="nccl",
            world_size=MPI.COMM_WORLD.Get_size(),
            rank=MPI.COMM_WORLD.Get_rank(),
            init_method=init_method,
        )

        local_rank = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)


