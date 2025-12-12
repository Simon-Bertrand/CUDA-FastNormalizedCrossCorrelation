#pragma once
#include <torch/extension.h>

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
