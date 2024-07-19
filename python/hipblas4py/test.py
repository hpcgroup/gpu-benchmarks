from build_kernels import build
import torch
import time

if __name__ == "__main__":
    build()
    import pyhipblas
    pyhipblas.init_handle()
    A = torch.rand(2048, 6144, device='cuda', dtype=torch.bfloat16)
    B = torch.rand(6144, 12288, device='cuda', dtype=torch.bfloat16)
    C = torch.zeros(2048, 12288, device='cuda', dtype=torch.bfloat16)
    pyhipblas.matmul_AB(A, B, C)
    print(C)
    exit()

    
