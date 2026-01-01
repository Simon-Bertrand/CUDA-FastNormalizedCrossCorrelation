
import torch
import pytest
import warnings
from hypothesis import strategies as st, given, settings
from .utils import naive_zncc
from torch_cireg import fft_zncc

@settings(max_examples=50, deadline=None)
@given(
    device=st.sampled_from(["cpu", "cuda"]),
    dtype=st.sampled_from([torch.float32, torch.float64]),
    B=st.integers(min_value=1, max_value=4),
    C=st.integers(min_value=1, max_value=4),
    H=st.integers(min_value=32, max_value=64),
    W=st.integers(min_value=32, max_value=64),
    h_ratio=st.floats(min_value=0.2, max_value=0.5), 
    w_ratio=st.floats(min_value=0.2, max_value=0.5),
)
def test_zncc_randomized(device, dtype, B, C, H, W, h_ratio, w_ratio):
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available, skipping test")
        return 
        
    h = min(max(1, int(H * h_ratio)), H)
    w = min(max(1, int(W * w_ratio)), W)

    eps = 1e-5

    torch.manual_seed(42)
    image = torch.randn(B, C, H, W, dtype=dtype, device=device, requires_grad=True)
    kernel = torch.randn(B, C, h, w, dtype=dtype, device=device, requires_grad=True)

    try:
        # 1. Forward
        out = fft_zncc(image, kernel, eps)
        ref = naive_zncc(image, kernel, eps)

        # Range check [-1, 1] with tolerance
        max_val = out.max().item()
        min_val = out.min().item()
        assert max_val <= 1.0 + 1e-4, f"Output max {max_val} > 1.0"
        assert min_val >= -1.0 - 1e-4, f"Output min {min_val} < -1.0"
        
        tolerance = {torch.float32: 1e-2, torch.float64: 1e-3}[dtype]
        
        diff = (out.detach() - ref).abs().max().item()
        # Debug print if failure likely
        if diff >= tolerance:
             print(f"FAILED: Diff {diff}, OutRange [{out.min():.4f}, {out.max():.4f}], RefRange [{ref.min():.4f}, {ref.max():.4f}]")
        
        assert diff < tolerance, f"Forward correctness failed! Diff: {diff}"
        
        # 2. Backward
        grad_output = torch.randn_like(out)
        g_img_opt, g_ker_opt = torch.autograd.grad(out, (image, kernel), grad_output, retain_graph=True)
        g_img_ref, g_ker_ref = torch.autograd.grad(ref, (image, kernel), grad_output, retain_graph=True)
        
        # Image Grad
        diff_img = (g_img_opt - g_img_ref).abs().max().item()
        assert diff_img < tolerance , f"Image Grad failed! Diff: {diff_img}"
        
        # Kernel Grad
        if g_ker_opt is not None:
            diff_ker = (g_ker_opt - g_ker_ref).abs().max().item()
            assert diff_ker < tolerance , f"Kernel Grad failed! Diff: {diff_ker}"

    except RuntimeError as e:
        pytest.fail(f"Runtime Error: {e}")
