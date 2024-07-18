import torch

torch.ops.load_library('gemm_wrapper.so')

if __name__ == '__main__':
    a = torch.ones(2, 3, device='cuda').half()
    b = torch.ones(3, 4, device='cuda').half()
    c = torch.ones(2, 4, device='cuda').half()
    d = torch.ones(2, 4, device='cuda').half()

    torch.ops.gemm_wrapper.rocblas_gemm_ex_wrapper(0, 0, 2, 4, 3, a, 2, b, 3, c, 2, d, 2)
