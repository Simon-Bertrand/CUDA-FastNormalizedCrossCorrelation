import torch
import numpy as np
from scipy import signal




def naive_cc(image : torch.Tensor, kernel : torch.Tensor):
    return (image.unfold(-2, kernel.shape[-2],1).unfold(-2, kernel.shape[-1],1) * kernel.unsqueeze(-3).unsqueeze(-3)).sum((-2,-1))
