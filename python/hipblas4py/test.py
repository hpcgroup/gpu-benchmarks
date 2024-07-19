from build_kernels import build
import torch
import pytest
build()
# you can only import pyhipblas after build has been called
import pyhipblas
pyhipblas.init_handle()

@pytest.mark.parametrize("m", [2048, 4096, 8192, 16384])
@pytest.mark.parametrize("k", [2048, 4096, 5120, 7168, 8192, 9216, 12288])
@pytest.mark.parametrize("n", [2048, 4096, 5120, 7168, 8192, 9216, 12288])
def test_AB(m, k, n):
    A = torch.randn(m,k, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(k,n, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(m,n, device="cuda", dtype=torch.bfloat16)
    pyhipblas.matmul_AB(A, B, C)
    C_pt = torch.matmul(A, B)
    assert torch.allclose(C.float(), C_pt.float())


@pytest.mark.parametrize("m", [2048, 4096, 8192, 16384])
@pytest.mark.parametrize("k", [2048, 4096, 5120, 7168, 8192, 9216, 12288])
@pytest.mark.parametrize("n", [2048, 4096, 5120, 7168, 8192, 9216, 12288])
def test_ABT(m, k, n):
    A = torch.randn(m,k, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(n,k, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(m,n, device="cuda", dtype=torch.bfloat16)
    pyhipblas.matmul_ABT(A, B, C)
    C_pt = torch.matmul(A, B.t())
    assert torch.allclose(C.float(), C_pt.float())


@pytest.mark.parametrize("m", [2048, 4096, 8192, 16384])
@pytest.mark.parametrize("k", [2048, 4096, 5120, 7168, 8192, 9216, 12288])
@pytest.mark.parametrize("n", [2048, 4096, 5120, 7168, 8192, 9216, 12288])
def test_ATB(m, k, n):
    A = torch.randn(k,m, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(k,n, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(m,n, device="cuda", dtype=torch.bfloat16)
    pyhipblas.matmul_ATB(A, B, C)
    C_pt = torch.matmul(A.t(), B)
    assert torch.allclose(C.float(), C_pt.float())

if __name__ == "__main__":
    test_AB(2048, 4096, 16384)
    test_ABT(2048, 4096, 16384)
    test_ATB(2048, 4096, 16384)

    
