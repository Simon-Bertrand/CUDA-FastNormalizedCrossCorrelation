import torch
import numpy as np
from scipy import signal




def naive_cc(image : torch.Tensor, kernel : torch.Tensor):
    return (image.unfold(-2, kernel.shape[-2],1).unfold(-2, kernel.shape[-1],1) * kernel.unsqueeze(-3).unsqueeze(-3)).sum((-2,-1))


def naive_local_std(image : torch.Tensor, ones_kernel : torch.Tensor, eps : float = 1e-5):
    return image.unfold(-2, ones_kernel.shape[-2],1).unfold(-2, ones_kernel.shape[-1],1).std(dim=[-2,-1], unbiased=False)

def naive_zncc(image: torch.Tensor, kernel: torch.Tensor, eps: float = 1e-5):
    # 1. Normalize Kernel
    k_centered = kernel - kernel.mean(dim=(-2, -1), keepdim=True)
    k_std = torch.norm(k_centered, p=2, dim=(-2, -1), keepdim=True)
    k_norm = k_centered / (k_std + eps)
    
    # 2. Local Std
    ones = torch.ones_like(kernel)
    std_map = naive_local_std(image, ones, eps)
    
    # 3. Correlation
    cc = naive_cc(image, k_norm)
    
    # 4. ZNCC
    N = kernel.shape[-2] * kernel.shape[-1]
    return cc / ((std_map + eps) * np.sqrt(N))







    