Here is information from Claude about conda environment to install pytorch and other other libraries for more performant batched matrix multiplication 
that is compatible with Apple Silicon:

# Create a new conda environment
conda create -n matrix_perf python=3.10 -y

# Activate the environment
conda activate matrix_perf

# Install base scientific computing packages
conda install -c conda-forge \
    numpy \
    scipy \
    pandas \
    matplotlib \
    -y

# Install PyTorch with MPS support
# Use the latest version compatible with Apple Silicon
pip install torch torchvision torchaudio

# Install JAX with CPU support (Apple Silicon)
pip install "jax[cpu]"

# Install additional performance libraries
pip install \
    numba \
    cupy-cuda11x \
    tensorflow \
    pytest-benchmark

import torch
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Initialized:", torch.backends.mps.is_built())

# Recommended libraries
pip install torch  # Primary GPU acceleration
pip install numpy  # CPU computations
pip install jax    # CPU computations

Then see batch_mult.py for an example of batched matrix multiplications that could be applied to accelerate TMM calculations!
