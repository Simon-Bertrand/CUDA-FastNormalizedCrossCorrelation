#include <torch/extension.h>
#include <vector>
#include <cmath>
#include "local_std.h"
#include "fft_cc.h"

// =========================================================================
// CUDA IMPLEMENTATION
// =========================================================================

std::vector<torch::Tensor> local_std_forward_cuda(torch::Tensor image, torch::Tensor ones_kernel, double eps) {
    auto B = image.size(0);
    auto h = ones_kernel.size(-2);
    auto w = ones_kernel.size(-1);
    double N = (double)(h * w);

    // 1. Prepare Inputs
    auto image_sq = image.square();
    auto input_stack = torch::cat({image, image_sq}, 0);

    // Kernel Handling
    torch::Tensor kernel_arg;
    int64_t k_el = ones_kernel.numel();
    int64_t k_flat = k_el / (h * w);

    if (k_flat == 1) {
        kernel_arg = ones_kernel;
    } else {
        kernel_arg = torch::cat({ones_kernel, ones_kernel}, 0);
    }

    // 2. Compute Sums via FFT
    auto sums = fft_cc_forward_cuda(input_stack, kernel_arg);
    
    // 3. Split (Views)
    auto chunks = sums.chunk(2, 0);
    auto s1 = chunks[0];
    auto s2 = chunks[1];

    // 4. Compute Std (Using native torch ops on CUDA for simplicity/speed)
    // var = (s2 - s1*s1/N) / N
    // std = sqrt(var + eps)

    auto var = (s2 - (s1.square()) / N) / N;
    var = torch::relu(var);
    auto out = torch::sqrt(var + eps);

    return {out, s1.clone()};
}


// CUDA Backward
torch::Tensor local_std_backward_cuda(torch::Tensor grad_output, torch::Tensor image, torch::Tensor ones_kernel, torch::Tensor std_map, torch::Tensor s1, double eps) {
    auto h = ones_kernel.size(-2);
    auto w = ones_kernel.size(-1);
    double N = (double)(h * w);

    // 1. Compute Grads (dS1, dS2)
    auto denom = 2.0 * std_map;
    auto d_var = grad_output / (denom + 1e-12);

    auto d_s2 = d_var / N;
    auto d_s1 = d_var * (-2.0 * s1 / (N*N));

    auto grad_stack = torch::cat({d_s1, d_s2}, 0); // [2B, ...]

    // 2. Convolution Backward
    // Kernel Handling
    torch::Tensor kernel_arg;
    int64_t k_el = ones_kernel.numel();
    int64_t k_flat = k_el / (h * w);

    if (k_flat == 1) {
        kernel_arg = ones_kernel;
    } else {
        kernel_arg = torch::cat({ones_kernel, ones_kernel}, 0);
    }

    // Call fft_cc_backward with mask {true, false} (only dI)
    auto d_inputs_vec = fft_cc_backward_cuda(grad_stack, torch::Tensor(), kernel_arg, {true, false});
    auto d_inputs = d_inputs_vec[0]; // [2B, ...]

    // 3. Combine
    // dI = dI_s1 + 2*I*dI_s2
    auto d_chunks = d_inputs.chunk(2, 0);
    auto d_i_s1 = d_chunks[0];
    auto d_i_s2 = d_chunks[1];

    auto d_image = d_i_s1 + 2.0 * image * d_i_s2;
    
    return d_image;
}
