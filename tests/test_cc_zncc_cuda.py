
import torch
import pytest
import numpy as np
from torch_cireg import fft_cross_correlation
from .utils import naive_cc

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("img_size, kernel_size", [
    ((64, 64), (64, 64)),   # Full size kernel
    ((32, 32), (1, 1)),     # 1x1 kernel
    ((81, 122), (17, 109)), # Odd/Even mix
    ((50, 132), (21, 89)),
    ((104, 117), (26, 5)),
    ((82, 31), (32, 8)),    # Kernel larger in one dim? PyTorch conv supports this, we assume kernel <= image
    ((93, 89), (23, 35)),
    ((51, 137), (27, 51)),
    ((88, 71), (62, 17)),
    ((91, 76), (64, 53)),
    ((145, 93), (133, 53)),
    ((50, 102), (41, 20)),
    ((89, 43), (11, 28)),
    ((31, 113), (17, 62)),
    ((73, 37), (49, 5)),
    ((110, 65), (52, 42)),
    ((35, 83), (12, 6)),
    ((122, 92), (20, 46)),
    ((103, 91), (16, 50)),
    ((101, 107), (89, 64)),
    ((109, 138), (84, 113)),
    ((53, 55), (27, 47)),
    ((58, 44), (47, 3)),
    ((100, 38), (90, 3)),
    ((40, 144), (19, 138)),
    ((64, 62), (61, 7)),
    ((132, 70), (30, 9)),
    ((101, 41), (36, 35)),
    ((52, 91), (26, 39)),
    ((73, 133), (37, 67)),
    ((130, 76), (80, 5)),
    ((34, 119), (25, 16)),
    ((56, 38), (17, 17)),
    ((71, 106), (53, 65)),
    ((142, 81), (98, 6)),
    ((130, 52), (105, 47)),
    ((72, 58), (38, 15)),
    ((88, 144), (30, 68)),
    ((74, 91), (59, 8)),
    ((91, 104), (64, 99)),
    ((56, 91), (15, 5)),
    ((99, 101), (29, 11))
])
def test_standard_cc_correctness_cuda(dtype, img_size, kernel_size):
    H, W = img_size
    h, w = kernel_size

    if h > H or w > W:
        pytest.skip("Kernel larger than image not supported")

    torch.manual_seed(42)
    image = torch.randn(1, H, W, dtype=dtype, device="cuda")
    kernel = torch.randn(1, h, w, dtype=dtype, device="cuda")

    out = fft_cross_correlation(image, kernel, normalize=False)
    ref = naive_cc(image, kernel, normalize=False)

    diff = (out.cpu() - ref).abs().max().item()

    tol = 1e-4 if dtype == torch.float32 else 1e-12
    if dtype == torch.float32: tol = 1e-3

    assert diff < tol, f"Max diff: {diff}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("img_size, kernel_size", [
    ((64, 64), (64, 64)),
    ((32, 32), (1, 1)),
    ((81, 122), (17, 109)),
    ((50, 132), (21, 89)),
    ((104, 117), (26, 5)),
    ((82, 31), (32, 8)),
    ((93, 89), (23, 35)),
    ((51, 137), (27, 51)),
    ((88, 71), (62, 17)),
    ((91, 76), (64, 53)),
    ((145, 93), (133, 53)),
    ((50, 102), (41, 20)),
    ((89, 43), (11, 28)),
    ((31, 113), (17, 62)),
    ((73, 37), (49, 5)),
    ((110, 65), (52, 42)),
    ((35, 83), (12, 6)),
    ((122, 92), (20, 46)),
    ((103, 91), (16, 50)),
    ((101, 107), (89, 64)),
    ((109, 138), (84, 113)),
    ((53, 55), (27, 47)),
    ((58, 44), (47, 3)),
    ((100, 38), (90, 3)),
    ((40, 144), (19, 138)),
    ((64, 62), (61, 7)),
    ((132, 70), (30, 9)),
    ((101, 41), (36, 35)),
    ((52, 91), (26, 39)),
    ((73, 133), (37, 67)),
    ((130, 76), (80, 5)),
    ((34, 119), (25, 16)),
    ((56, 38), (17, 17)),
    ((71, 106), (53, 65)),
    ((142, 81), (98, 6)),
    ((130, 52), (105, 47)),
    ((72, 58), (38, 15)),
    ((88, 144), (30, 68)),
    ((74, 91), (59, 8)),
    ((91, 104), (64, 99)),
    ((56, 91), (15, 5)),
    ((99, 101), (29, 11))
])
def test_zncc_correctness_cuda(dtype, img_size, kernel_size):
    H, W = img_size
    h, w = kernel_size

    if h > H or w > W:
        pytest.skip("Kernel larger than image")

    torch.manual_seed(42)
    image = torch.randn(1, H, W, dtype=dtype, device="cuda")
    kernel = torch.randn(1, h, w, dtype=dtype, device="cuda")

    out = fft_cross_correlation(image, kernel, normalize=True)
    ref = naive_cc(image, kernel, normalize=True)

    diff = (out.cpu() - ref).abs().max().item()

    tol = 1e-4 if dtype == torch.float32 else 1e-7

    assert diff < tol, f"Max diff: {diff}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_high_dimensionality_support_cuda(dtype):
    # Test batching with extra dims
    B, C, H, W = 2, 3, 64, 64
    h, w = 16, 16
    
    image = torch.randn(B, C, H, W, dtype=dtype, device="cuda")
    kernel = torch.randn(B, C, h, w, dtype=dtype, device="cuda")
    
    out = fft_cross_correlation(image, kernel, normalize=True)
    
    assert out.shape == (B, C, H - h + 1, W - w + 1)
    
    slice_out = out[0, 0]
    slice_ref = naive_cc(image[0, 0:1], kernel[0, 0:1], normalize=True).squeeze()

    diff = (slice_out.cpu() - slice_ref).abs().max().item()
    assert diff < 1e-4

def test_error_handling_cuda():
    img = torch.randn(1, 16, 16, device="cuda")
    ker = torch.randn(1, 32, 32, device="cuda")
    
    with pytest.raises(RuntimeError):
        fft_cross_correlation(img, ker)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_class_api_cuda(dtype):
    from torch_cireg import VeryFastNormalizedCrossCorrelation

    H, W = 64, 64
    h, w = 16, 16
    img = torch.randn(1, H, W, dtype=dtype, device="cuda")
    ker = torch.randn(1, h, w, dtype=dtype, device="cuda")

    model = VeryFastNormalizedCrossCorrelation(H, W, h, w, normalize=True)
    out = model.forward(img, ker)

    assert out.shape == (1, H - h + 1, W - w + 1)
