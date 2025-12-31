
import torch
import pytest
import time
from hypothesis import strategies as st, given, settings
from .utils import naive_cc
from torch_cireg import fft_cc

@settings(max_examples=50, deadline=None)
@given(
    device=st.sampled_from(["cpu", "cuda"]),
    dtype=st.sampled_from([torch.float32, torch.float64]),
    B=st.integers(min_value=1, max_value=4),
    C=st.integers(min_value=1, max_value=4),
    H=st.integers(min_value=32, max_value=128),
    W=st.integers(min_value=32, max_value=128),
    h_ratio=st.floats(min_value=0.1, max_value=0.9),
    w_ratio=st.floats(min_value=0.1, max_value=0.9),
)
def test_cc_randomized(device, dtype, B, C, H, W, h_ratio, w_ratio):
    if device == "cuda" and not torch.cuda.is_available():
        return 
    h = min(max(1, int(H * h_ratio)), H)
    w = min(max(1, int(W * w_ratio)), W)

    torch.manual_seed(42)
    image = torch.randn(B, C, H, W, dtype=dtype, device=device, requires_grad=True)
    kernel = torch.randn(B, C, h, w, dtype=dtype, device=device, requires_grad=True)

    try:
        out = fft_cc(image, kernel)
        assert out.device == image.device, f"Output device {out.device} does not match input device {image.device}"
        ref = naive_cc(image, kernel)
        tolerance = {torch.float32: 5e-3, torch.float64: 1e-4}[dtype]
        diff = (out.detach() - ref).abs().max().item()
        assert diff < tolerance, f"Correctness failed! Diff: {diff}, Shape: {image.shape} vs {kernel.shape}, Device: {device}, Dtype: {dtype}"
        
        # Backward Pass Check against Naive Autograd
        grad_output = torch.randn_like(out)
        
        # Gradients for FFT implementation
        g_image_fft, g_kernel_fft = torch.autograd.grad(out, (image, kernel), grad_output, retain_graph=True)
        
        # Gradients for Naive implementation
        g_image_ref, g_kernel_ref = torch.autograd.grad(ref, (image, kernel), grad_output, retain_graph=True)
        
        diff_g_image = (g_image_fft - g_image_ref).abs().max().item()
        diff_g_kernel = (g_kernel_fft - g_kernel_ref).abs().max().item()
        
        assert diff_g_image < tolerance, f"Backward Image Grad failed! Diff: {diff_g_image}"
        assert diff_g_kernel < tolerance, f"Backward Kernel Grad failed! Diff: {diff_g_kernel}"

        if dtype == torch.float64 and H < 40 and W < 40:
             assert torch.autograd.gradcheck(fft_cc, (image, kernel), eps=1e-6, atol=1e-4)

    except RuntimeError as e:
        pytest.fail(f"Runtime Error: {e}")



@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_speed_benchmark(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Use reasonably large sizes to ensure FFT is faster
    B, C = 2, 2
    H, W = 256, 256
    h, w = 32, 32
    
    # Setup Data
    image = torch.randn(B, C, H, W, dtype=dtype, device=device, requires_grad=True)
    kernel = torch.randn(B, C, h, w, dtype=dtype, device=device, requires_grad=True)
    
    grad_output = torch.randn(B, C, H - h + 1, W - w + 1, dtype=dtype, device=device)

    # Warmup
    for _ in range(3):
        out = fft_cc(image, kernel)
        out.backward(grad_output, retain_graph=True)
        image.grad.zero_()
        kernel.grad.zero_()
    
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark FFT
    start_time = time.time()
    iterations = 10
    for _ in range(iterations):
        out = fft_cc(image, kernel)
        out.backward(grad_output, retain_graph=True)
        # Ensure async C++ and CUDA calls are scheduled
        if device == "cuda":
             torch.cuda.synchronize()
        else:
             # Basic CPU sync relies on Python overhead, usually sufficient for this granularity
             pass
             
    if device == "cuda":
        torch.cuda.synchronize()
        
    fft_duration = (time.time() - start_time) / iterations

    # Benchmark Naive
    # Naive is much slower, run fewer iterations
    start_time = time.time()
    naive_iterations = 2 
    for _ in range(naive_iterations):
        ref = naive_cc(image, kernel)
        ref.backward(grad_output, retain_graph=True)
        if device == "cuda":
             torch.cuda.synchronize()

    if device == "cuda":
        torch.cuda.synchronize()

    naive_duration = (time.time() - start_time) / naive_iterations

    print(f"\n[Speed Test] Device: {device}, Dtype: {dtype}")
    print(f"FFT Time:   {fft_duration:.6f} s")
    print(f"Naive Time: {naive_duration:.6f} s")
    print(f"Speedup:    {naive_duration / fft_duration:.2f}x")


    expected_speedup_factor = 0.015 if device == "cuda" else 0.05

    assert fft_duration < expected_speedup_factor*naive_duration, f"FFT implementation is slower! ({fft_duration:.6f} vs {naive_duration:.6f})"
