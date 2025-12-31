# Torch CiReg: Fast FFT Cross-Correlation for PyTorch

This repository contains a high-performance **PyTorch C++/CUDA Extension** for computing 2D Cross-Correlation using Fast Fourier Transforms (FFT).

It is designed to be significantly faster than naive spatial implementations for large kernels and images, utilizing **cuFFT** for GPU acceleration and optimizations like batched R2C transforms and minimized memory allocations.

## Features

- **Standard Cross-Correlation**: efficiently computes $O = I \star K$.
- **Dual Device Support**:
  - **CUDA**: Optimized custom kernels using `cuFFT` with single contiguous memory allocation for high throughput.
  - **CPU**: Fallback implementation using multithreaded operations (via PyTorch/CPU ops).
- **Full Autograd Support**:
  - Custom C++ `torch::autograd::Function` implementation.
  - Differentiable with respect to both **Image** and **Kernel**.
  - Backward pass rigorously verified against naive spatial implementations.
- **Precision**: Supports `float32` and `float64` (Double precision).
- **Correctness**: Verified with property-based testing (`hypothesis`) across various shapes, batch sizes, and datatypes.

## Installation

**Prerequisites**:
- Python 3.x
- PyTorch (>= 1.7)
- CUDA Toolkit (if building with GPU support)
- C++17 compatible compiler (MSVC on Windows, GCC/Clang on Linux)

### Build and Install

```bash
pip install .
```

To build in editable mode for development:

```bash
pip install -e . --no-build-isolation
```

## Usage

The library exposes a simple functional API `fft_cc`.

```python
import torch
from torch_cireg import fft_cc

# Define input tensors
# Shape: [Batch, Channels, Height, Width]
B, C, H, W = 1, 1, 1024, 1024
h, w = 128, 128

image = torch.randn(B, C, H, W, device='cuda', requires_grad=True)
kernel = torch.randn(B, C, h, w, device='cuda', requires_grad=True)

# Compute Cross-Correlation
output = fft_cc(image, kernel)

# Dimensions: [Batch, H - h + 1, W - w + 1] (Valid padding)
print(f"Output shape: {output.shape}") 

# Compute Gradients
loss = output.sum()
loss.backward()

print(f"Image Grad: {image.grad.shape}")
print(f"Kernel Grad: {kernel.grad.shape}")
```

## Testing

The project uses `pytest` and `hypothesis` for robust testing.

```bash
pytest
```

Tests cover:
- CPU and CUDA implementations.
- Correctness vs Naive spatial loop implementation.
- Gradient checks (Autograd).
- Variable batch sizes and aspect ratios.
- Edge cases (small inputs).

## Implementation Details

The core computation relies on the convolution theorem:
$$ I \star K = \mathcal{F}^{-1}(\mathcal{F}(I) \cdot \overline{\mathcal{F}(K)}) $$

For the backward pass (gradients), we implement efficient analytic gradients using convolution/correlation transposition rules, batched into a single FFT execution context to minimize launch overheads and memory fragmentation.

See [GRADIENTS.md](GRADIENTS.md) for the mathematical derivation of the gradients.
