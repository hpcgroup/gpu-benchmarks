## hipblas4py : A python wrapper around hipblas/rocblas

### Correctness tests
Install pytest first

```
pip install pytest
```


Then run

```
python -m pytest test.py
```

This will JIT compile the extension, run a test suite of matmuls and compare the outputs with those of `torch.matmul`.


### Benchmarking

WIP
