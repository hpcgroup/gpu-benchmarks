from build_kernels import build
import torch

if __name__ == "__main__":
    build()
    import pyhipblas
    tensor = torch.tensor([2,3], device='cuda')
    pyhipblas.print_tensor(tensor)
    pyhipblas.init_handle()
    exit()
    A = torch.tensor([1,2,3,4], device='cuda', dtype=torch.bfloat16).reshape(2,2)
    B = torch.tensor([1,2,3,4], device='cuda', dtype=torch.bfloat16).reshape(2,2)
    C = torch.zeros(2,2, device='cuda', dtype=torch.bfloat16)
    pyhipblas.matmul_AB(A, B, C)
    print(C)

    
