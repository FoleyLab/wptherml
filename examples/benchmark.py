import numpy as np
import timeit
from scipy.linalg.blas import zgemm
from numba import jit

# Try to import CuPy (GPU-accelerated NumPy)
try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cupy_available = False

# Try to import PyTorch (GPU Tensor acceleration)
try:
    import torch
    torch_available = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch_available = False

# Matrix size and batch size for benchmarking
matrix_size = 2  # Keep matrices small (2x2) for now
batch_size = 10**6  # 1 million matrix multiplications in parallel

# Generate random complex matrices for CPU (NumPy/SciPy)
np.random.seed(42)
A_batch = np.random.rand(batch_size, matrix_size, matrix_size) + 1j * np.random.rand(batch_size, matrix_size, matrix_size)
B_batch = np.random.rand(batch_size, matrix_size, matrix_size) + 1j * np.random.rand(batch_size, matrix_size, matrix_size)

# Generate CuPy GPU matrices if available
if cupy_available:
    A_gpu = cp.asarray(A_batch)
    B_gpu = cp.asarray(B_batch)

# Generate PyTorch GPU matrices if available
if torch_available:
    A_torch = torch.tensor(A_batch, dtype=torch.complex128, device=device)
    B_torch = torch.tensor(B_batch, dtype=torch.complex128, device=device)

# Numba JIT-optimized batched multiplication
@jit(nopython=True, parallel=True)
def numba_batched_matmul(A, B):
    result = np.empty_like(A)
    for i in range(A.shape[0]):
        result[i] = A[i] @ B[i]
    return result

# Define benchmark functions
def numpy_batched_matmul():
    return A_batch @ B_batch  # NumPy batched multiplication

def blas_batched_zgemm():
    return np.array([zgemm(1.0, A_batch[i], B_batch[i]) for i in range(batch_size)])  # SciPy BLAS batched multiplication

def numba_batched_matmul_test():
    return numba_batched_matmul(A_batch, B_batch)  # Numba JIT batched multiplication

if cupy_available:
    def cupy_batched_matmul():
        return A_gpu @ B_gpu  # CuPy GPU batched multiplication

if torch_available:
    def torch_batched_matmul():
        return torch.matmul(A_torch, B_torch)  # PyTorch GPU batched multiplication

# Measure execution times
numpy_time = timeit.timeit(numpy_batched_matmul, number=1)
blas_time = timeit.timeit(blas_batched_zgemm, number=1)
numba_time = timeit.timeit(numba_batched_matmul_test, number=1)

# GPU benchmarks (only if CuPy or PyTorch are available)
if cupy_available:
    cupy_time = timeit.timeit(cupy_batched_matmul, number=1)

if torch_available:
    torch_time = timeit.timeit(torch_batched_matmul, number=1)

# Print results
print(f"\nBenchmarking batched {matrix_size}x{matrix_size} complex matrix multiplications ({batch_size} iterations):\n")
print(f"NumPy batched @ operator:    {numpy_time:.6f} seconds")
print(f"Numba JIT batched @ operator: {numba_time:.6f} seconds  ({numpy_time / numba_time:.2f}x faster than NumPy)")
print(f"SciPy BLAS batched zgemm:    {blas_time:.6f} seconds  ({numpy_time / blas_time:.2f}x faster than NumPy)")

if cupy_available:
    print(f"CuPy GPU batched @ operator: {cupy_time:.6f} seconds  ({numpy_time / cupy_time:.2f}x faster than NumPy)")

if torch_available:
    print(f"PyTorch GPU batched @ operator: {torch_time:.6f} seconds  ({numpy_time / torch_time:.2f}x faster than NumPy)")

print("\nNote: GPU acceleration is significant for large batch sizes (100,000+ matrices).")

