import torch
import time
import numpy as np

def naive_sequential_multiply(matrices):
    """
    Naive sequential multiplication approach
    
    Parameters:
    matrices (torch.Tensor): Shape [outer_dim, inner_dim, 2, 2]
    
    Returns:
    torch.Tensor: Resulting products, shape [outer_dim, 2, 2]
    """
    outer_dim, inner_dim, _, _ = matrices.shape
    products = torch.zeros(outer_dim, 2, 2, dtype=matrices.dtype, device=matrices.device)
    
    for i in range(outer_dim):
        result = matrices[i, 0, :, :]
        for j in range(1, inner_dim):
            result = torch.matmul(result, matrices[i, j, :, :])
        products[i] = result
    
    return products

def vectorized_multiply(matrices):
    """
    Vectorized multiplication approach
    
    Parameters:
    matrices (torch.Tensor): Shape [outer_dim, inner_dim, 2, 2]
    
    Returns:
    torch.Tensor: Resulting products, shape [outer_dim, 2, 2]
    """
    outer_dim, inner_dim, _, _ = matrices.shape
    # Reshape to combine first two dimensions
    reshaped = matrices.reshape(-1, inner_dim, 2, 2)
    
    # Initialize result tensor
    products = torch.zeros(reshaped.shape[0], 2, 2, dtype=matrices.dtype, device=matrices.device)
    
    # Compute initial product
    products = reshaped[:, 0, :, :]
    
    # Perform batched matrix multiplication
    for j in range(1, inner_dim):
        products = torch.bmm(products, reshaped[:, j, :, :])
    
    return products.reshape(outer_dim, 2, 2)

def benchmark_matrix_multiplication(outer_dim=20000, inner_dim=500):
    """
    Benchmark different multiplication approaches
    
    Parameters:
    outer_dim (int): Number of sets of matrices to multiply
    inner_dim (int): Number of matrices to multiply in each set
    """
    # Prepare device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Generate random matrices
    matrices = torch.rand(outer_dim, inner_dim, 2, 2, device=device)
    
    # Benchmark naive approach
    start = time.time()
    naive_result = naive_sequential_multiply(matrices)
    torch.mps.synchronize()
    naive_time = time.time() - start
    print(f"Naive Sequential Approach Time: {naive_time:.4f} seconds")
    
    # Benchmark vectorized approach
    start = time.time()
    vectorized_result = vectorized_multiply(matrices)
    torch.mps.synchronize()
    vectorized_time = time.time() - start
    print(f"Vectorized Approach Time: {vectorized_time:.4f} seconds")
    
    # Verify results are close
    print("\nResult Verification:")
    print("Max Absolute Difference:", 
          torch.max(torch.abs(naive_result - vectorized_result)).item())
    
    # Performance comparison
    speedup = naive_time / vectorized_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    return naive_result, vectorized_result

# Diagnostic and usage
if __name__ == "__main__":
    print("Matrix Multiplication Benchmark on Apple Silicon")
    print("Checking MPS Availability:", torch.backends.mps.is_available())
    
    # Run benchmark
    benchmark_matrix_multiplication(outer_dim=1000, inner_dim=4)

    benchmark_matrix_multiplication(outer_dim=2000, inner_dim=4)
    benchmark_matrix_multiplication(outer_dim=2000, inner_dim=8)
    benchmark_matrix_multiplication(outer_dim=20000, inner_dim=100)
