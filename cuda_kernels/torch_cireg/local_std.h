#pragma once
#include <torch/extension.h>
#include <vector>

// Forward Declaration
// Returns: {StdMap, S1} - We return S1 so it can be saved for backward
std::vector<torch::Tensor> local_std_forward_cpu(torch::Tensor image, torch::Tensor ones_kernel, double eps);

// Backward Declaration
// Inputs: grad_output, image, ones_kernel, std_map, s1 (saved from forward)
torch::Tensor local_std_backward_cpu(torch::Tensor grad_output, torch::Tensor image, torch::Tensor ones_kernel, torch::Tensor std_map, torch::Tensor s1, double eps);

#ifdef WITH_CUDA
// Returns {StdMap, S1}
std::vector<torch::Tensor> local_std_forward_cuda(torch::Tensor image, torch::Tensor ones_kernel, double eps);
torch::Tensor local_std_backward_cuda(torch::Tensor grad_output, torch::Tensor image, torch::Tensor ones_kernel, torch::Tensor std_map, torch::Tensor s1, double eps);
#endif
