
import torch
import pytest
import numpy as np
from torch_cireg import fft_cc_forward, fft_cc_backward
from .utils import naive_cc

class CCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, kernel):
        ctx.save_for_backward(image, kernel)
        return fft_cc_forward(image, kernel)

    @staticmethod
    def backward(ctx, grad_output):
        image, kernel = ctx.saved_tensors
        # fft_cc_backward returns std::vector<Tensor> -> list of Tensors
        grads = fft_cc_backward(grad_output.contiguous(), image, kernel)
        return grads[0], grads[1]

def fft_cc(image, kernel):
    return CCFunction.apply(image, kernel)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("img_size, kernel_size", [
    ((64, 64), (64, 64)),   # Full size kernel
    ((32, 32), (1, 1)),     # 1x1 kernel
    ((81, 122), (17, 109)), # Odd/Even mix
    ((50, 132), (21, 89)),
    ((104, 117), (26, 5)),
])
def test_cc_correctness_cpu(dtype, img_size, kernel_size):
    H, W = img_size
    h, w = kernel_size

    if h > H or w > W:
        pytest.skip("Kernel larger than image not supported")

    torch.manual_seed(42)
    image = torch.randn(1, H, W, dtype=dtype, requires_grad=True)
    kernel = torch.randn(1, h, w, dtype=dtype, requires_grad=True)

    out = fft_cc(image, kernel)
    ref = naive_cc(image, kernel, normalize=False)

    diff = (out.detach() - ref).abs().max().item()

    tol = 1e-4 if dtype == torch.float32 else 2e-5
    assert diff < tol, f"Max diff: {diff}"

    # Backward Check
    # We use gradcheck
    # gradcheck requires double precision for reliability
    if dtype == torch.float64:
        assert torch.autograd.gradcheck(fft_cc, (image, kernel), eps=1e-6, atol=1e-4)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_high_dimensionality_support_cpu(dtype):
    # Test batching with extra dims
    B, C, H, W = 2, 3, 64, 64
    h, w = 16, 16

    image = torch.randn(B, C, H, W, dtype=dtype)
    kernel = torch.randn(B, C, h, w, dtype=dtype)

    out = fft_cc_forward(image, kernel)

    # Check shape
    assert out.shape == (B, C, H - h + 1, W - w + 1)
