#pragma once
#include <torch/extension.h>
#include <vector>

// Forward Declaration
// Returns: {ZNCC, StdMap, S1, K_Norm}
// ZNCC: The result
// StdMap: Saved for backward
// S1: Saved for local_std backward
// K_Norm: Saved for CC backward
std::vector<torch::Tensor> fft_zncc_forward_cpu(torch::Tensor image, torch::Tensor kernel, double eps);

// Backward Declaration
// Returns: {grad_image, grad_kernel}
std::vector<torch::Tensor> fft_zncc_backward_cpu(
    torch::Tensor grad_output, 
    torch::Tensor image, 
    torch::Tensor kernel, 
    torch::Tensor std_map, 
    torch::Tensor s1, 
    torch::Tensor k_norm, 
    double eps);

#ifdef WITH_CUDA
std::vector<torch::Tensor> fft_zncc_forward_cuda(torch::Tensor image, torch::Tensor kernel, double eps);
std::vector<torch::Tensor> fft_zncc_backward_cuda(
    torch::Tensor grad_output, 
    torch::Tensor image, 
    torch::Tensor kernel, 
    torch::Tensor std_map, 
    torch::Tensor s1, 
    torch::Tensor k_norm, 
    double eps);
#endif
