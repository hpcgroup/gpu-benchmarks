### NCCL4Py: Python bindings that are an interface to NCCL

I am developing this because PyTorch's interface to NCCL while being user friendly is not flexible enough for low-level usecases

## Requirements
Requires PyTorch to be installed.

## Compiling

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ../
make -j 
```


### Example - 
```
import torch
torch.ops.load_library('build/libnccl4py.so')

## initialize torch.distributed first 

## form a communication group with size 3 
## for example if there are 6 GPUs, then GPUs [0,1,2] will form a group
## and GPUs [3,4,5] will form another group
## this will also initialize a dedicated stream for communication
torch.ops.nccl4py.setup_communicator_and_stream(3)
	
## get rank, world-size and gpu stream
rank = torch.ops.nccl4py.get_world_rank()
size = torch.ops.nccl4py.get_world_size()
nccl_stream = torch.cuda.ExternalStream(torch.ops.nccl4py.get_stream())

x = torch.ones(2,3, device='cuda').half()

## issues an asynchronous non-blocking call to NCCL-Allreduce
torch.ops.nccl4py.all_reduce_fp16(x)
torch.cuda.synchronize()
print(x) ## should be all 3s

```


