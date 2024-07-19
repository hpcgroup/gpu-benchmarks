from build_kernels import build
import torch
import pytest
build()
# you can only import pyhipblas after build has been called
import pyhipblas
pyhipblas.init_handle()

def benchmark(m, k, n, warmup=5, iters=10, format_="AB"):
    assert format_ in ["AB", "ABT", "ATB"]
    if format_ == "AB":
        A = torch.randn(m,k, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(k,n, device="cuda", dtype=torch.bfloat16)
    elif format_ == "ABT":
        A = torch.randn(m,k, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(n,k, device="cuda", dtype=torch.bfloat16)
    elif format_ == "ATB":
        A = torch.randn(k,m, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(k,n, device="cuda", dtype=torch.bfloat16)

    tflops = 2 * m * n * k / 1e12
    
    # warmup
    for _ in range(warmup):
        C = torch.zeros(m,n, device="cuda", dtype=torch.bfloat16)
        if format_ == "AB":
            pyhipblas.matmul_AB(A, B, C)
            torch.matmul(A, B)
        elif format_ == "ABT":
            pyhipblas.matmul_ABT(A, B, C)
            torch.matmul(A, B.t())
        elif format_ == "ATB":
            pyhipblas.matmul_ATB(A, B, C)
            torch.matmul(A.t(), B)
    
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        C = torch.zeros(m,n, device="cuda", dtype=torch.bfloat16)
        if format_ == "AB":
            pyhipblas.matmul_AB(A, B, C)
        elif format_ == "ABT":
            pyhipblas.matmul_ABT(A, B, C)
        elif format_ == "ATB":
            pyhipblas.matmul_ATB(A, B, C)
    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end)/1000
    tflops_s = tflops*iters/time

    print(f"Pyhipblas - TFLOP/s = {tflops_s:.2f} TFLOP/s | Perc of Peak = {tflops_s/192*100:.2f} %")


    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        if format_ == "AB":
            torch.matmul(A, B)
        elif format_ == "ABT":
            torch.matmul(A, B.t())
        elif format_ == "ATB":
            torch.matmul(A.t(), B)

    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end)/1000
    tflops_s = tflops*iters/time

    print(f"Pytorch - TFLOP/s = {tflops_s:.2f} TFLOP/s | Perc of Peak = {tflops_s/192*100:.2f} %")


if __name__ == "__main__":
    H = 12288
    B = 1
    S = 2048
    print(f"m={B*S} k={H} n={3*H//2}")
    benchmark(B*S, H, 3*H//2, format_="ABT")


    print(f"m={B*S} k={H//2} n={H}")
    benchmark(B*S, H//2, H, format_="ABT")


    print(f"m={B*S} k={H} n={4*H//2}")
    benchmark(B*S, H, 4*H//2, format_="ABT")
    

    print(f"m={B*S} k={4*H//2} n={H}")
    benchmark(B*S, 4*H//2, H, format_="ABT")
