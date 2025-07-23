import time
import torch
import cupy as cp
import numpy as np

# -----------------------
#  PyTorch implementations
# -----------------------

def torch_naive(matrices: torch.Tensor) -> torch.Tensor:
    outer, inner, _, _ = matrices.shape
    products = torch.zeros(outer, 2, 2, dtype=matrices.dtype, device=matrices.device)
    for i in range(outer):
        result = matrices[i, 0]
        for j in range(1, inner):
            result = result @ matrices[i, j]
        products[i] = result
    return products

def torch_batched(matrices: torch.Tensor) -> torch.Tensor:
    outer, inner, _, _ = matrices.shape
    # start with the first “column”
    products = matrices[:, 0]
    for j in range(1, inner):
        products = torch.bmm(products, matrices[:, j])
    return products

# -----------------------
#  CuPy implementations
# -----------------------

def cupy_naive(matrices: cp.ndarray) -> cp.ndarray:
    outer, inner, _, _ = matrices.shape
    products = cp.zeros((outer, 2, 2), dtype=matrices.dtype)
    for i in range(outer):
        result = matrices[i, 0]
        for j in range(1, inner):
            result = result.dot(matrices[i, j])
        products[i] = result
    return products

def cupy_batched(matrices: cp.ndarray) -> cp.ndarray:
    # leverages elementwise “@” for batch matmul
    outer, inner, _, _ = matrices.shape
    products = matrices[:, 0]
    for j in range(1, inner):
        # this dispatches a single kernel over the batch each time
        products = products @ matrices[:, j]
    return products

# -----------------------
#  Benchmark harness
# -----------------------

def benchmark_all(outer=2000, inner=8, dtype=torch.float32):
    print(f"\n=== Benchmark: outer={outer}, inner={inner} ===")

    # --- PyTorch setup (CPU or MPS/GPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else
                          "cpu")
    t_mats = torch.rand(outer, inner, 2, 2, dtype=dtype, device=device)

    # Torch naive
    torch.cuda.synchronize() if device.type=="cuda" else None
    start = time.time()
    tn = torch_naive(t_mats)
    torch.cuda.synchronize() if device.type=="cuda" else None
    t_naive_torch = time.time() - start
    print(f"Torch naive   : {t_naive_torch:.4f}s")

    # Torch batched
    torch.cuda.synchronize() if device.type=="cuda" else None
    start = time.time()
    tb = torch_batched(t_mats)
    torch.cuda.synchronize() if device.type=="cuda" else None
    t_batch_torch = time.time() - start
    print(f"Torch batched : {t_batch_torch:.4f}s  (speedup {t_naive_torch/t_batch_torch:.1f}×)")

    # verify correctness
    diff = (tn - tb).abs().max().item()
    print(f" Torch Δ max   : {diff:.3e}")

    # --- CuPy setup ---
    # generate random complex or real arrays on GPU
    c_dtype = np.complex64 if dtype==torch.complex64 else np.float32
    real = cp.random.randn(outer, inner, 2, 2, dtype=c_dtype)
    if c_dtype==np.complex64:
        mats = real + 1j*cp.random.randn(outer, inner, 2, 2, dtype=c_dtype)
    else:
        mats = real

    cp.cuda.Device().synchronize()
    # CuPy naive
    start = time.time()
    cn = cupy_naive(mats)
    cp.cuda.Device().synchronize()
    t_naive_cupy = time.time() - start
    print(f" CuPy naive   : {t_naive_cupy:.4f}s")

    # CuPy batched
    start = time.time()
    cb = cupy_batched(mats)
    cp.cuda.Device().synchronize()
    t_batch_cupy = time.time() - start
    print(f" CuPy batched : {t_batch_cupy:.4f}s  (speedup {t_naive_cupy/t_batch_cupy:.1f}×)")

    # verify correctness
    diff_c = cp.max(cp.abs(cn - cb)).item()
    print(f" CuPy Δ max   : {diff_c:.3e}")

if __name__ == "__main__":
    # Try the same dims you used in your torch tests:
    for (o, i) in [(1000,4), (2000,4), (2000,8), (20000,100)]:
        benchmark_all(outer=o, inner=i)

