#include <torch/extension.h>
#include <vector>
#include <cmath>
#include "fft_zncc.h"
#include "local_std.h"
#include "fft_cc.h"

// Helper for Global Kernel Normalization
// Returns {K_Norm}
torch::Tensor normalize_kernel_cpu(torch::Tensor kernel, double eps) {
    auto centered = kernel - kernel.mean({-2, -1}, true);
    auto std = torch::norm(centered, 2, {-2, -1}, true);
    return centered / (std + eps);
}

// =========================================================================
// CPU IMPLEMENTATION
// =========================================================================

std::vector<torch::Tensor> fft_zncc_forward_cpu(torch::Tensor image, torch::Tensor kernel, double eps) {
    // 1. Normalize Kernel (Global)
    auto k_norm = normalize_kernel_cpu(kernel, eps);
    
    // 2. Local Std
    // We need 'ones' kernel. Construct it on the fly.
    auto ones = torch::ones_like(kernel);
    auto std_res = local_std_forward_cpu(image, ones, eps);
    auto std_map = std_res[0];
    auto s1 = std_res[1];

    // 3. Cross Correlation (I, K_Norm)
    auto cc = fft_cc_forward_cpu(image, k_norm);

    // 4. ZNCC = CC / StdMap
    // Note: Dimensions must align. 
    // StdMap is (B, C, H-h+1, W-w+1) same as CC.
    // 4. ZNCC = CC / (StdMap * sqrt(N))
    double N = (double)(kernel.size(-2) * kernel.size(-1));
    auto zncc = cc / ((std_map + eps) * std::sqrt(N));

    return {zncc, std_map, s1, k_norm};
}

std::vector<torch::Tensor> fft_zncc_backward_cpu(
    torch::Tensor grad_output, 
    torch::Tensor image, 
    torch::Tensor kernel, 
    torch::Tensor std_map, 
    torch::Tensor s1, 
    torch::Tensor k_norm, 
    double eps) {
    
    // ZNCC = CC / Std  (Let Std = (std_map + eps) * sqrt(N))
    
    auto std_eps = std_map + eps;
    auto ones = torch::ones_like(kernel);
    double N = (double)(kernel.size(-2) * kernel.size(-1));
    double sqrt_N = std::sqrt(N);

    // 1. Gradient w.r.t CC
    // dL/dCC = 1 / (StdMap * sqrt(N))
    auto d_cc = grad_output / (std_eps * sqrt_N);

    // 2. Gradient w.r.t StdMap
    // dL/dStd = -ZNCC / Std
    auto cc = fft_cc_forward_cpu(image, k_norm);
    auto d_std = -d_cc * cc / std_eps;

    // 3. Backprop through CC
    auto d_cc_inputs = fft_cc_backward_cpu(d_cc, image, k_norm);
    auto d_image_cc = d_cc_inputs[0];
    auto d_k_norm = d_cc_inputs[1];

    // 4. Backprop through Local Std
    auto d_image_std = local_std_backward_cpu(d_std, image, ones, std_map, s1, eps);

    // 5. Total Image Gradient
    auto d_image = d_image_cc + d_image_std;

    // 6. Backprop through Kernel Normalization
    auto k_mean = kernel.mean({-2, -1}, true);
    auto k_centered = kernel - k_mean;
    auto k_sigma = torch::norm(k_centered, 2, {-2, -1}, true);
    auto k_sigma_eps = k_sigma + eps;

    auto dot = (d_k_norm * k_norm).sum({-2, -1}, true);
    auto d_k0 = (d_k_norm - k_norm * dot) / k_sigma_eps;
    auto d_kernel = d_k0 - d_k0.mean({-2, -1}, true);

    return {d_image, d_kernel};
}
