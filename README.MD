# Torch CiReg: Fast FFT ZNCC & Cross-Correlation for PyTorch

This repository contains a high-performance **PyTorch C++/CUDA Extension** for computing **Zero-Normalized Cross-Correlation (ZNCC)** and standard Cross-Correlation using Fast Fourier Transforms (FFT).

It allows for extremely fast image template matching and registration tasks, offering significant speedups over naive spatial implementations (especially for large kernels) while maintaining full differentiability.

## Features

- **Zero-Normalized Cross-Correlation (ZNCC)**:
  - Robust matching invariant to local brightness and contrast changes.
  - Efficiently computes:
    $$ZNCC(I, K) = \frac{(I - \mu_{local}) \cdot (K - \mu_{global}) / \sigma_{global}}{\sigma_{local} \cdot \sqrt{N}}$$
  - Uses a **Fused CUDA Kernel** to compute local standard deviation maps rapidly.
  
- **Standard Cross-Correlation**:
  - Efficient FFT-based computation: $O = I \star K$.
  
- **Performance Optimized**:
  - **CUDA**: Custom kernels utilizing `cuFFT` with batched R2C transforms and in-place operations to minimize memory usage.
  - **Fused Kernels**: Local variance calculation is fused into single passes to reduce global memory checks.
  - **CPU Fallback**: Multithreaded implementations for non-CUDA environments.

- **Full Autograd Support**:
  - Differentiable with respect to both **Image** and **Kernel**.
  - Essential for deep learning based registration or feature tracking pipelines.
  - Gradients analytically derived and verified against numerical differentiation.

- **Precision & Reliability**: 
  - Supports `float32` and `float64`.
  - Rigorously tested with `hypothesis` property-based testing.

## Installation

**Prerequisites**:
- Python 3.8+
- PyTorch (>= 1.7)
- CUDA Toolkit (11.x+)
- C++17 compatible compiler

### Build from Source

```bash
# Install with pip
pip install .

# For development (editable mode)
pip install -e . --no-build-isolation
```

## Usage

### Zero-Normalized Cross-Correlation (ZNCC)

This is the primary operator for robust template matching.

```python
import torch
from torch_cireg import fft_zncc

# Define input tensors [Batch, Channels, Height, Width]
B, C, H, W = 1, 1, 1024, 1024
h, w = 128, 128

image = torch.randn(B, C, H, W, device='cuda', requires_grad=True)
kernel = torch.randn(B, C, h, w, device='cuda', requires_grad=True)

# Compute ZNCC
# Output values are in range [-1, 1]
output = fft_zncc(image, kernel)

print(f"ZNCC Output shape: {output.shape}") # [B, H-h+1, W-w+1]

# Backpropagation
loss = (1 - output.max()).mean()
loss.backward()

print(f"Image Grad: {image.grad.shape}")
print(f"Kernel Grad: {kernel.grad.shape}")
```

### Standard Cross-Correlation & Local Std

You can also access the low-level operators directly:

```python
from torch_cireg import fft_cc, local_std

# 1. Standard FFT Cross-Correlation
cc = fft_cc(image, kernel)

# 2. Local Standard Deviation Map
# Computes std dev of 'image' within windows of size 'kernel'
ones_kernel = torch.ones_like(kernel)
std_map = local_std(image, ones_kernel)
```

## Architecture

The implementation is split into a Python wrapper and C++/CUDA backend:

- **`torch_cireg` (Python)**: Main entry point and type checking.
- **`cuda_kernels` (C++/CUDA)**:
  - **`fft_cc`**: Core FFT convolution logic using cuFFT `PlanMany`.
  - **`local_std`**: Methods to compute local windowed variance using mean-of-squares minus square-of-means approach, optimized with fused kernels.
  - **`fft_zncc`**: orchestrator that chains normalization and correlation.

## Testing

Run the test suite to verify installation and correctness:

```bash
# Run all tests
pytest

# Run generic benchmarks
pytest tests/test_cc.py tests/test_local_std.py
```

## Documentation

- [GRADIENTS.md](GRADIENTS.md): Derivations of gradients for Cross-Correlation.
- [NORMALIZATION.md](NORMALIZATION.md): Detail on the ZNCC formula and local standard deviation derivatives.
