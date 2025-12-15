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

torch::Tensor fft_cross_correlation_cpu(
    torch::Tensor image,
    torch::Tensor kernel,
    bool normalize
);

#ifdef WITH_CUDA
torch::Tensor fft_cross_correlation_cuda(
    torch::Tensor image,
    torch::Tensor kernel,
    bool normalize
);

// Class API helpers
void* create_cuda_impl(int img_h, int img_w, int ker_h, int ker_w, bool normalize);
void destroy_cuda_impl(void* impl);
torch::Tensor forward_cuda_impl(void* impl, torch::Tensor image, torch::Tensor kernel);
#endif
