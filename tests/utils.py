
import torch
import numpy as np
from scipy import signal

def naive_cc(image, kernel, normalize=False):
    """
    Naive cross-correlation using only torch, unfold, and no FFT. Batched.
    Args:
        image: [B, H, W] torch.Tensor or numpy.ndarray
        kernel: [B, h, w] torch.Tensor or numpy.ndarray
    Returns:
        out: [B, H-h+1, W-w+1] torch.Tensor
    """
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image)
    if not isinstance(kernel, torch.Tensor):
        kernel = torch.from_numpy(kernel)
    # Ensure float
    image = image.float()
    kernel = kernel.float()

    B, H, W = image.shape
    _, h, w = kernel.shape
    out_h = H - h + 1
    out_w = W - w + 1

    # Unfold image to all sliding windows
    # Shape: [B, h*w, out_h*out_w]
    image_unf = torch.nn.functional.unfold(image.unsqueeze(1), (h, w)).view(B, h*w, out_h*out_w)

    # Unfold kernel for each batch, flatten kernel to (B, h*w, 1)
    kernel_unf = kernel.view(B, 1, h*w).transpose(1,2)  # (B, h*w, 1)

    # (B, h*w, out_h*out_w) * (B, h*w, 1) -> (B, out_h*out_w)
    out = (image_unf * kernel_unf).sum(dim=1)

    # Reshape to (B, out_h, out_w)
    out = out.view(B, out_h, out_w)
    return out.cpu()
