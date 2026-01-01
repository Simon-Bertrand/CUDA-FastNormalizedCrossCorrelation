#include <torch/extension.h>
#include <vector>
#include <cmath>
#include "fft_zncc.h"
#include "local_std.h"
#include "fft_cc.h"

// =========================================================================
// CUDA IMPLEMENTATION
// =========================================================================

// Helpers
torch::Tensor normalize_kernel_cuda(torch::Tensor kernel, double eps) {
    auto mean = kernel.mean({-2, -1}, true);
    auto centered = kernel - mean;
    auto std = torch::norm(centered, 2, {-2, -1}, true);
    return centered / (std + eps);
}

std::vector<torch::Tensor> fft_zncc_forward_cuda(torch::Tensor image, torch::Tensor kernel, double eps) {
    // 1. Normalize Kernel
    auto k_norm = normalize_kernel_cuda(kernel, eps);
    
    // 2. Local Std
    auto ones = torch::ones_like(kernel);
    auto std_res = local_std_forward_cuda(image, ones, eps);
    auto std_map = std_res[0];
    auto s1 = std_res[1];

    // 3. CC
    auto cc = fft_cc_forward_cuda(image, k_norm);

    // 4. ZNCC = CC / (Std * sqrt(N))
    // we need N from kernel size
    double N = (double)(kernel.size(-2) * kernel.size(-1));
    auto zncc = cc / ((std_map + eps) * std::sqrt(N));

    return {zncc, std_map, s1, k_norm};
}

std::vector<torch::Tensor> fft_zncc_backward_cuda(
    torch::Tensor grad_output, 
    torch::Tensor image, 
    torch::Tensor kernel, 
    torch::Tensor std_map, 
    torch::Tensor s1, 
    torch::Tensor k_norm, 
    double eps) {

    auto std_eps = std_map + eps;
    auto ones = torch::ones_like(kernel);
    double N = (double)(kernel.size(-2) * kernel.size(-1));
    double sqrt_N = std::sqrt(N);

    // 1. Grads Intermediate
    // dZNCC/dCC = 1 / (Std * sqrt(N))
    auto d_cc = grad_output / (std_eps * sqrt_N);

    // dZNCC/dStd = -ZNCC / Std
    auto cc = fft_cc_forward_cuda(image, k_norm);
    auto d_std = -d_cc * cc / std_eps;

    // 2. Backprop CC
    auto d_cc_inputs = fft_cc_backward_cuda(d_cc, image, k_norm);
    auto d_image_cc = d_cc_inputs[0];
    auto d_k_norm = d_cc_inputs[1];

    // 3. Backprop Std
    auto d_image_std = local_std_backward_cuda(d_std, image, ones, std_map, s1, eps);

    // 4. Total Image
    auto d_image = d_image_cc + d_image_std;

    // 5. Backprop Kernel
    auto k_mean = kernel.mean({-2, -1}, true);
    auto k_centered = kernel - k_mean;
    auto k_sigma = torch::norm(k_centered, 2, {-2, -1}, true);
    auto k_sigma_eps = k_sigma + eps;

    auto dot = (d_k_norm * k_norm).sum({-2, -1}, true);
    auto d_k0 = (d_k_norm - k_norm * dot) / k_sigma_eps;
    auto d_kernel = d_k0 - d_k0.mean({-2, -1}, true);

    return {d_image, d_kernel}; 
}
