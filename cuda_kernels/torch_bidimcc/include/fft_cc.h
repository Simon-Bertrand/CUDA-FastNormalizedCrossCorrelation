#pragma once
#include <torch/extension.h>
#include <vector>

// Helper: Reshape Output
inline std::vector<int64_t> change_h_w_shapes(const torch::Tensor& tensor, const int64_t& H, const int64_t& W) {
    std::vector<int64_t> out;
    out.reserve(tensor.dim());
    out.insert(out.end(), tensor.sizes().begin(), tensor.sizes().end() - 2);
    out.insert(out.end(), {H,W});
    return out;
}

// Forward Pass: Returns Output
torch::Tensor fft_cc_forward_cpu(torch::Tensor image, torch::Tensor kernel);

// Backward Pass: Returns {grad_image, grad_kernel}
// output_mask: [compute_d_image, compute_d_kernel]
std::vector<torch::Tensor> fft_cc_backward_cpu(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel, std::vector<bool> output_mask = {true, true});

#ifdef WITH_CUDA
torch::Tensor fft_cc_forward_cuda(torch::Tensor image, torch::Tensor kernel);
std::vector<torch::Tensor> fft_cc_backward_cuda(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel, std::vector<bool> output_mask = {true, true});
#endif
