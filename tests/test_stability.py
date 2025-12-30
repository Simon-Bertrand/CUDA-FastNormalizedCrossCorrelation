
import torch
import pytest
import numpy as np
from torch_cireg import fft_cross_correlation

def test_zncc_stability_cpu():
    """
    Verifies that ZNCC is stable against large DC offsets in the image.
    ZNCC should be invariant to adding a constant to the image.
    """
    H, W = 128, 128
    h, w = 32, 32

    # Random image and kernel
    torch.manual_seed(42)
    image = torch.randn(1, H, W, dtype=torch.float32)
    kernel = torch.randn(1, h, w, dtype=torch.float32)

    # 1. Standard ZNCC
    out_ref = fft_cross_correlation(image, kernel, normalize=True)

    # 2. Large Offset ZNCC
    # Add a large offset. For float32, 1e5 is significant enough to cause
    # loss of precision in variance calculation (x^2 term) if not centered.
    # (1e5)^2 = 1e10. float32 has ~7 decimal digits.
    # If variance is small (e.g. 1), 1e10 + 1 - 1e10 = 0 (catastrophic cancellation).
    offset = 1e6
    image_offset = image + offset

    out_offset = fft_cross_correlation(image_offset, kernel, normalize=True)

    # Compare
    # We expect them to be very close.
    # Without centering, this fails for float32.
    diff = (out_ref - out_offset).abs().max().item()
    print(f"Max diff with offset {offset}: {diff}")

    # We accept a small error due to precision, but it should not be garbage.
    assert diff < 1e-3, f"ZNCC instability detected! Max diff: {diff}"

def test_zncc_stability_float64_cpu():
    """
    Verifies stability for float64.
    Double precision usually handles 1e6 fine, but we test 1e12 to stress it or just ensure consistency.
    """
    H, W = 128, 128
    h, w = 32, 32

    torch.manual_seed(42)
    image = torch.randn(1, H, W, dtype=torch.float64)
    kernel = torch.randn(1, h, w, dtype=torch.float64)

    out_ref = fft_cross_correlation(image, kernel, normalize=True)

    offset = 1e9
    image_offset = image + offset

    out_offset = fft_cross_correlation(image_offset, kernel, normalize=True)

    diff = (out_ref - out_offset).abs().max().item()
    print(f"Max diff with offset {offset} (double): {diff}")

    assert diff < 1e-7, f"ZNCC instability detected for double! Max diff: {diff}"

if __name__ == "__main__":
    test_zncc_stability_cpu()
    test_zncc_stability_float64_cpu()
