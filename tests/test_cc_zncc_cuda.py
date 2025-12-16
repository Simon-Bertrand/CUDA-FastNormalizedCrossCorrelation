import pytest
import torch
import torch.nn.functional as F
import numpy as np
import random

# ==============================================================================
# 1. SETUP & GROUND TRUTH IMPLEMENTATIONS
# ==============================================================================
try:
    import torch_cireg
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

@pytest.fixture(scope="module")
def device():
    # User requested to adapt for CUDA testing without running on actual GPU if not available.
    # However, to test the CUDA implementation, we must call the function with CUDA tensors.
    # If no CUDA device is present, we cannot truly "run" the test.
    # But since the goal is to have the file ready for CUDA testing, we default to 'cuda'.
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA extension not installed or no GPU available")
    return torch.device('cuda')

def naive_cross_correlation(image, kernel):
    """Standard Naive Cross Correlation (Unnormalized)."""
    # Flatten batch dimensions for consistency with C++ kernel logic
    orig_shape = image.shape
    H, W = orig_shape[-2:]
    h, w = kernel.shape[-2:]
    
    img_flat = image.view(-1, 1, H, W)
    ker_flat = kernel.view(-1, 1, h, w)
    
    unfolded = img_flat.unfold(-2, h, 1).unfold(-2, w, 1)
    result = (unfolded * ker_flat.unsqueeze(-3).unsqueeze(-3)).sum((-2, -1))
    
    # Reshape back to original batch dims
    out_h, out_w = result.shape[-2:]
    return result.view(*orig_shape[:-2], out_h, out_w)

def naive_zncc(image, kernel):
    """Naive ZNCC implementation using Cosine Similarity."""
    orig_shape = image.shape
    H, W = orig_shape[-2:]
    h, w = kernel.shape[-2:]
    
    img_flat = image.view(-1, 1, H, W)
    ker_flat = kernel.view(-1, 1, h, w)

    # Unfold and Center Window
    windows = img_flat.unfold(-2, h, 1).unfold(-2, w, 1)
    win_mean = windows.mean(dim=(-2, -1), keepdim=True)
    win_centered = windows - win_mean
    
    # Center Kernel
    ker_mean = ker_flat.mean(dim=(-2, -1), keepdim=True)
    ker_centered = ker_flat - ker_mean
    
    # Flatten spatial dims for Cosine Sim
    flat_win = win_centered.flatten(-2, -1)
    flat_ker = ker_centered.flatten(-2, -1).unsqueeze(-2).unsqueeze(-2)
    
    # Cosine Sim
    res = F.cosine_similarity(flat_win, flat_ker, dim=-1)
    
    # Reshape back
    out_h, out_w = res.shape[-2:]
    return res.view(*orig_shape[:-2], out_h, out_w)

# ==============================================================================
# 2. TEST GENERATORS (Odd/Even/Rectangular coverage)
# ==============================================================================
def generate_shapes(n=40):
    cases = []
    # 1. Edge Case: Kernel = Image (1x1 output)
    cases.append((1, 64, 64, 64, 64))
    # 2. Edge Case: 1x1 Kernel (Copy)
    cases.append((1, 32, 32, 1, 1))
    
    for _ in range(n):
        B = np.random.randint(1, 4)
        # Random parity for Image and Kernel to ensure robust FFT padding tests
        H = np.random.randint(30, 150)
        W = np.random.randint(30, 150)
        h = np.random.randint(3, H - 2)
        w = np.random.randint(3, W - 2)
        cases.append((B, H, W, h, w))
    return cases

TEST_PARAMS = generate_shapes()
dtypes = [torch.float32, torch.float64]

# ==============================================================================
# 3. TESTS
# ==============================================================================

@pytest.mark.parametrize("dtype", dtypes)
def test_identity_alignment_cuda(device, dtype):
    """
    Verifies that a delta kernel (1 in center) produces an output perfectly 
    aligned with the input image. Crucial for FFT padding correctness.
    """
    H, W = 32, 32
    h, w = 3, 3 
    img = torch.randn(1, H, W, device=device, dtype=dtype)
    
    # Delta Kernel
    ker = torch.zeros(1, h, w, device=device, dtype=dtype)
    ker[0, 1, 1] = 1.0 
    
    # Run Standard CC
    res_cuda = torch_cireg.fft_cross_correlation(img, ker, False)
    
    # Valid output size is 30x30. 
    # Output[0,0] corresponds to window Img[0:3, 0:3] dot Kernel.
    # Kernel has 1 at [1,1]. So dot product = Img[1,1].
    expected_val = img[0, 1, 1].item()
    actual_val = res_cuda[0, 0, 0].item()
    
    assert abs(expected_val - actual_val) < 1e-4, \
        f"Alignment Error: Expected {expected_val}, Got {actual_val}"

@pytest.mark.parametrize("B, H, W, h, w", TEST_PARAMS)
@pytest.mark.parametrize("dtype", dtypes)
def test_standard_cc_correctness_cuda(device, B, H, W, h, w, dtype):
    """Verifies Standard Cross Correlation against Naive implementation."""
    img = torch.rand(B, H, W, device=device, dtype=dtype)*2-0.5
    ker = torch.rand(B, h, w, device=device, dtype=dtype)*2-0.5
    
    res_cuda = torch_cireg.fft_cross_correlation(img, ker, False)
    res_ref = naive_cross_correlation(img, ker)
    
    # Shape Check
    assert res_cuda.shape == res_ref.shape
    
    # Value Check (Relative Error for robustness with large sums)
    diff = (res_cuda - res_ref).abs()
    # Adding epsilon to denominator to handle near-zero values
    rel_err = diff / (res_ref.abs() + 1e-3)
    max_rel_err = rel_err.max().item()
    
    # 1% relative error tolerance for float32 FFT vs Spatial
    tol = 0.01 if dtype == torch.float32 else 1e-4
    assert max_rel_err < tol, f"StdCC Mismatch (Max RelErr: {max_rel_err:.4f})"

@pytest.mark.parametrize("B, H, W, h, w", TEST_PARAMS)
@pytest.mark.parametrize("dtype", dtypes)
def test_zncc_correctness_cuda(device, B, H, W, h, w, dtype):
    """Verifies ZNCC against Naive implementation."""
    img = torch.rand(B, H, W, device=device, dtype=dtype)*2-0.5
    ker = torch.rand(B, h, w, device=device, dtype=dtype)*2-0.5
    # Ensure kernel variance != 0 to avoid NaNs in ground truth
    ker.view(B, -1)[:, 0] += 1.0 
    
    res_cuda = torch_cireg.fft_cross_correlation(img, ker, True)
    res_ref = naive_zncc(img, ker)
    
    assert res_cuda.shape == res_ref.shape
    assert not torch.isnan(res_cuda).any()
    
    # Value Check (Mean Absolute Error)
    diff = (res_cuda - res_ref).abs()
    mae = diff.mean().item()
    max_diff = diff.max().item()
    
    # ZNCC involves squaring/sqrt, accumulating float32 errors.
    # MAE < 1e-4 is a strict pass for correctness.
    tol = 1e-4 if dtype == torch.float32 else 1e-7
    assert mae < tol, f"ZNCC Mismatch: MAE={mae:.8f}, MaxDiff={max_diff:.8f}"

@pytest.mark.parametrize("dtype", dtypes)
def test_high_dimensionality_support_cuda(device, dtype):
    """
    Verifies support for ND tensors (e.g. 5D).
    The kernel should flatten all leading dims into a single batch.
    """
    # Shape: (Batch=2, Time=2, Channel=2, H=32, W=32) -> Effective Batch = 8
    dims = (2, 2, 2, 32, 32)
    k_dims = (2, 2, 2, 5, 5)
    
    img = torch.randn(*dims, device=device, dtype=dtype)
    ker = torch.randn(*k_dims, device=device, dtype=dtype)
    
    res = torch_cireg.fft_cross_correlation(img, ker, False)
    
    expected_shape = (2, 2, 2, 32-5+1, 32-5+1)
    assert res.shape == expected_shape, f"ND Shape Mismatch: Got {res.shape}"
    assert not torch.isnan(res).any()

def test_error_handling_cuda(device):
    """Verifies that invalid input shapes raise errors."""
    # 1D Tensor (too few dims)
    img = torch.randn(10, device=device)
    ker = torch.randn(3, device=device)
    
    # Should catch either in Python wrapper or C++ TORCH_CHECK
    with pytest.raises(RuntimeError):
        torch_cireg.fft_cross_correlation(img, ker, False)

# ==============================================================================
# 4. CLASS API TEST
# ==============================================================================
@pytest.mark.parametrize("dtype", dtypes)
def test_class_api_cuda(device, dtype):
    H, W = 64, 64
    h, w = 15, 15

    img = torch.randn(1, H, W, device=device, dtype=dtype)
    ker = torch.randn(1, h, w, device=device, dtype=dtype)

    # Initialize the VeryFastNormalizedCrossCorrelation class
    matcher = torch_cireg.VeryFastNormalizedCrossCorrelation(H, W, h, w, normalize=True)

    # Forward pass
    res = matcher.forward(img, ker)

    assert res is not None
    assert res.is_cuda
    assert res.shape == (1, H-h+1, W-w+1)
    assert res.dtype == dtype
