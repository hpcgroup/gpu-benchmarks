import torch
import torch.distributed as dist
import os
import time
import argparse
import sys

NUM_WARMUP_ITERATIONS = 5
DATA_TYPE = torch.bfloat16
DATA_TYPE_SIZE = 2

def initialize_data(size):
    data = torch.rand((size,), dtype=DATA_TYPE).cuda()
    return data

def benchmark_allgather(rank, world_size, num_gpus, min_msg_size, max_msg_size, iterations):
    print ("Initializing dist group", file=sys.stderr)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    local_data_size = max_msg_size
    global_data_size = local_data_size

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if rank == 0:
        print(f"Local data size: {local_data_size // (1024 * 1024)} MB", file=sys.stderr)
        print(f"Global data size: {global_data_size // (1024 * 1024)} MB", file=sys.stderr)

    for msg_size in [2**i for i in range(int(torch.log2(torch.tensor(min_msg_size))), int(torch.log2(torch.tensor(max_msg_size))) + 1)]:
        msg_count = msg_size // DATA_TYPE_SIZE

        local_data = initialize_data(msg_count) # Input
        global_data = initialize_data(msg_count // num_gpus) # Output

        # Warmup iterations
        for _ in range(NUM_WARMUP_ITERATIONS):
            dist.reduce_scatter_tensor(global_data, local_data)
            torch.cuda.synchronize()

        if msg_size >= 8388608:
            iterations = 20

        dist.barrier()
        total_time = 0.0
        for _ in range(iterations):
            start.record()
            dist.reduce_scatter_tensor(global_data, local_data)
            end.record()
            torch.cuda.synchronize()

            elapsed_time = start.elapsed_time(end)
            torch.cuda.synchronize()

            total_time+=elapsed_time / 1000

        dist.barrier()

        if rank == 0:
            print(f"{msg_size // 1024 // 1024},{msg_size / world_size / 1024 / 1024},{total_time / iterations:.6f}")

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='All-gather benchmark with PyTorch distributed')
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs')
    parser.add_argument('--min_msg_size', type=int, help='Minimum message size')
    parser.add_argument('--max_msg_size', type=int, help='Maximum message size')
    parser.add_argument('--iterations', type=int, help='Number of iterations')

    args = parser.parse_args()

    if args.num_gpus < 2 or args.min_msg_size <= 0 or args.max_msg_size <= 0 or args.min_msg_size > args.max_msg_size or args.iterations <= 0:
        print("Invalid input parameters.")
        exit(1)

    os.environ['WORLD_SIZE'] = str(args.num_gpus)
    os.environ['RANK'] = os.getenv('SLURM_PROCID', '0')

    rank = int(os.getenv('RANK'))
    world_size = int(os.getenv('WORLD_SIZE'))

    benchmark_allgather(rank, world_size, args.num_gpus, args.min_msg_size, args.max_msg_size, args.iterations)


