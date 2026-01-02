#pragma once
#include <torch/extension.h>
#include <vector>

// Forward Pass: Returns Output
std::vector<torch::Tensor> local_std_forward_cpu(torch::Tensor image, torch::Tensor ones, double eps);

// Backward Pass: Returns grad_image
torch::Tensor local_std_backward_cpu(torch::Tensor grad_output, torch::Tensor image, torch::Tensor ones, torch::Tensor std_map, torch::Tensor s1, double eps);

#ifdef WITH_CUDA
std::vector<torch::Tensor> local_std_forward_cuda(torch::Tensor image, torch::Tensor ones, double eps);
torch::Tensor local_std_backward_cuda(torch::Tensor grad_output, torch::Tensor image, torch::Tensor ones, torch::Tensor std_map, torch::Tensor s1, double eps);
#endif
