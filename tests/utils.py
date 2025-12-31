
import torch
import numpy as np
from scipy import signal

def naive_cc(image, kernel, normalize=False):
    """
    Naive implementation of CC/ZNCC using scipy.signal.fftconvolve for reference.
    This works on CPU (numpy).
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(kernel, torch.Tensor):
        kernel = kernel.detach().cpu().numpy()

    # Handle batch dimension by iterating
    # image: [B, H, W]
    # kernel: [B, h, w]

    B, H, W = image.shape
    _, h, w = kernel.shape

    out_list = []

    for b in range(B):
        I = image[b]
        K = kernel[b]

        if normalize:
            # ZNCC
            # Correlation = sum((I - mean_I)(K - mean_K)) / (std_I * std_K * N)

            # 1. Normalize Kernel
            K_mean = np.mean(K)
            K_norm = K - K_mean
            K_std = np.linalg.norm(K_norm)

            if K_std < 1e-6:
                out_list.append(np.zeros((H - h + 1, W - w + 1)))
                continue

            # 2. Compute numerator: sum(I * K_norm) - sum(I)*mean(K_norm)
            # Since mean(K_norm) is 0, Numerator = sum(I * K_norm)
            # We use correlation, so we flip kernel for convolution
            # But scipy.signal.correlate2d or fftconvolve with flipped kernel
            # fftconvolve(I, K_flipped, mode='valid') -> Correlation

            K_flipped = K_norm[::-1, ::-1]
            numerator = signal.fftconvolve(I, K_flipped, mode='valid')

            # 3. Compute Local Variance of Image
            # Var(I) = E[I^2] - (E[I])^2
            # Sums computed via convolution with ones
            ones = np.ones((h, w))
            N = h * w

            sum_I = signal.fftconvolve(I, ones, mode='valid')
            sum_I2 = signal.fftconvolve(I**2, ones, mode='valid')

            # Variance * N^2 = N * sum_I2 - sum_I^2
            var_term = N * sum_I2 - sum_I**2
            var_term = np.maximum(var_term, 0)

            local_std = np.sqrt(var_term) # This is N * sigma_I

            # Denominator = (N * sigma_I) * sigma_K / sqrt(N)?
            # Wait.
            # Denom = sqrt(sum(I_centered^2)) * sqrt(sum(K_centered^2))
            # sqrt(sum(I_centered^2)) = sqrt(N * sigma_I^2) = sqrt(N) * sigma_I
            # local_std calculated above is sqrt(N^2 * sigma_I^2) = N * sigma_I
            # So sqrt(sum(I_centered^2)) = local_std / sqrt(N)

            # Formula: Num / ( sqrt(sum(I^2)) * sqrt(sum(K^2)) )
            # = Num / ( (local_std / sqrt(N)) * K_std )
            # = (Num * sqrt(N)) / (local_std * K_std)

            denom = local_std * K_std

            # Handle division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                res = (numerator * np.sqrt(N)) / denom

            res[denom < 1e-5] = 0
            out_list.append(res)

        else:
            # Standard CC
            # Just correlation
            K_flipped = K[::-1, ::-1]
            res = signal.fftconvolve(I, K_flipped, mode='valid')
            out_list.append(res)

    return torch.tensor(np.stack(out_list), dtype=torch.float32)
