
import torch
import pytest
import time
from hypothesis import strategies as st, given, settings
from .utils import naive_local_std
from torch_cireg import local_std

@settings(max_examples=50, deadline=None)
@given(
    device=st.sampled_from(["cpu", "cuda"]),
    dtype=st.sampled_from([torch.float32, torch.float64]),
    B=st.integers(min_value=1, max_value=4),
    C=st.integers(min_value=1, max_value=4),
    H=st.integers(min_value=32, max_value=128),
    W=st.integers(min_value=32, max_value=128),
    h_ratio=st.floats(min_value=0.1, max_value=0.5), # Smaller windows for clearer variance
    w_ratio=st.floats(min_value=0.1, max_value=0.5),
)
def test_local_std_randomized(device, dtype, B, C, H, W, h_ratio, w_ratio):
    if device == "cuda" and not torch.cuda.is_available():
        return 
        
    h = min(max(1, int(H * h_ratio)), H)
    w = min(max(1, int(W * w_ratio)), W)

    # Use larger eps for stability during random testing, specifically for float32
    eps = 1e-5

    torch.manual_seed(42)
    # Ensure image range is reasonable to avoid massive squares
    image = torch.randn(B, C, H, W, dtype=dtype, device=device, requires_grad=True)
    
    # Kernel: Ones kernel as required by optimized implementation signature
    ones = torch.ones(B, C, h, w, dtype=dtype, device=device)

    try:
        # 1. Forward Check
        out = local_std(image, ones, eps)
        
        # Naive implementation
        ref = naive_local_std(image, ones, eps)
        
        # Compare
        # Note: Unfold method and FFT method might have slight numerical differences due to summing order
        # especially for sum of squares.
        tolerance = {torch.float32: 1e-3, torch.float64: 1e-4}[dtype]
        
        # Check alignment/valid size 
        # Optimized returns 'Valid' correlation size usually? 
        # Let's check output shape.
        # FFT CC returns 1..H-h+1. Naive Unfold returns 1..H-h+1.
        # Ensure sizes match
        assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
        
        diff = (out.detach() - ref).abs().max().item()
        assert diff < tolerance, f"Forward correctness failed! Diff: {diff}, Device: {device}, Dtype: {dtype}"
        
        # 2. Backward Check
        grad_output = torch.randn_like(out)
        
        # Optimized Grad
        g_image_opt = torch.autograd.grad(out, image, grad_output, retain_graph=True)[0]
        
        # Naive Grad
        g_image_ref = torch.autograd.grad(ref, image, grad_output, retain_graph=True)[0]
        
        diff_grad = (g_image_opt - g_image_ref).abs().max().item()
        # Gradients involving 1/sqrt(x) can be sensitive
        grad_tolerance = tolerance  
        
        assert diff_grad < grad_tolerance, f"Backward Grad failed! Diff: {diff_grad}"

    except RuntimeError as e:
        pytest.fail(f"Runtime Error: {e}")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_speed_benchmark(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Benchmarking Setup
    B, C = 2, 2
    H, W = 512, 512
    h, w = 32, 32  # Large kernel to stress test
    
    eps = 1e-5
    
    torch.manual_seed(42)
    image = torch.randn(B, C, H, W, dtype=dtype, device=device, requires_grad=True)
    ones = torch.ones(B, C, h, w, dtype=dtype, device=device)
    
    # Check valid output size for grad_output
    # Assuming valid padding: H - h + 1
    out_h = H - h + 1
    out_w = W - w + 1
    grad_output = torch.randn(B, C, out_h, out_w, dtype=dtype, device=device)

    # Warmup
    for _ in range(3):
        out = local_std(image, ones, eps)
        out.backward(grad_output, retain_graph=True)
        image.grad.zero_()
    
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark Optimized
    start_time = time.time()
    iterations = 20
    for _ in range(iterations):
        out = local_std(image, ones, eps)
        out.backward(grad_output, retain_graph=True)
        # Check if we need to zero grad if we are accumulating? 
        # But for speed test, accumulation is fine cost-wise.
        image.grad = None 
        
        if device == "cuda":
            torch.cuda.synchronize()
            
    if device == "cuda":
        torch.cuda.synchronize()
        
    optimized_duration = (time.time() - start_time) / iterations

    # Benchmark Naive
    # Naive unfold is very memory and compute heavy. Run fewer iters.
    naive_iterations = 5
    start_time = time.time()
    for _ in range(naive_iterations):
        ref = naive_local_std(image, ones, eps)
        ref.backward(grad_output, retain_graph=True)
        image.grad = None
        
        if device == "cuda":
            torch.cuda.synchronize()

    if device == "cuda":
        torch.cuda.synchronize()
        
    naive_duration = (time.time() - start_time) / naive_iterations

    print(f"\n[Speed Test LocalStd] Device: {device}, Dtype: {dtype}")
    print(f"Optimized Time: {optimized_duration:.6f} s")
    print(f"Naive Time:     {naive_duration:.6f} s")
    speedup = naive_duration / optimized_duration
    print(f"Speedup:        {speedup:.2f}x")

    # Expect significant speedup, especially on CUDA and large kernels
    # Naive is O(H*W*h*w), Optimized is O(H*W log(HW))
    # Threshold: Conservative 5x
    assert speedup > 2.0, f"Optimized implementation is not significantly faster! ({speedup:.2f}x)"
