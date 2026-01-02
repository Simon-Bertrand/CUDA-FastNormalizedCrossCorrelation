#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "local_std.h"
#include "fft_cc.h"

// ==================================================================================
// CUDA KERNELS
// ==================================================================================
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// 1. Forward Fused Kernel
template <typename scalar_t>
__global__ void fused_std_kernel(
    const scalar_t* __restrict__ s1,
    const scalar_t* __restrict__ s2,
    scalar_t* __restrict__ out,
    const scalar_t N_val,
    const scalar_t eps,
    const int numel) {

  CUDA_KERNEL_LOOP(index, numel) {
    scalar_t val_s1 = s1[index];
    scalar_t val_s2 = s2[index];

    // Variance = (S2 - S1^2/N) / N
    scalar_t var = (val_s2 - (val_s1 * val_s1) / N_val) / N_val;

    if (var < 0) var = 0;
    out[index] = sqrt(var + eps);
  }
}

// 2. Backward Prepare Kernel (dStd -> dS1, dS2)
template <typename scalar_t>
__global__ void fused_backward_prep_kernel(
    const scalar_t* __restrict__ grad_std,
    const scalar_t* __restrict__ std_map,
    const scalar_t* __restrict__ s1,
    scalar_t* __restrict__ grad_s1,
    scalar_t* __restrict__ grad_s2,
    const scalar_t N_val,
    const scalar_t eps,
    const int numel) {

  CUDA_KERNEL_LOOP(index, numel) {
    scalar_t g_std = grad_std[index];
    scalar_t std_val = std_map[index];
    scalar_t val_s1 = s1[index];

    scalar_t denom = static_cast<scalar_t>(2.0) * std_val;
    if (denom < 1e-12) denom = 1e-12; // Safety

    scalar_t d_var = g_std / denom;

    grad_s2[index] = d_var / N_val;
    grad_s1[index] = d_var * (static_cast<scalar_t>(-2.0) * val_s1 / (N_val * N_val));
  }
}

// 3. Backward Combine Kernel (dI_s1, dI_s2, I -> dI)
template <typename scalar_t>
__global__ void fused_backward_combine_kernel(
    const scalar_t* __restrict__ grad_i_s1,
    const scalar_t* __restrict__ grad_i_s2,
    const scalar_t* __restrict__ image,
    scalar_t* __restrict__ grad_input,
    const int numel) {

  CUDA_KERNEL_LOOP(index, numel) {
    scalar_t g1 = grad_i_s1[index];
    scalar_t g2 = grad_i_s2[index];
    scalar_t img = image[index];

    grad_input[index] = g1 + static_cast<scalar_t>(2.0) * img * g2;
  }
}

// ==================================================================================
// HOST FUNCTIONS
// ==================================================================================

std::vector<torch::Tensor> local_std_forward_cuda(torch::Tensor image, torch::Tensor ones_kernel, double eps) {
    auto B = image.size(0);
    auto h = ones_kernel.size(-2);
    auto w = ones_kernel.size(-1);
    double N = (double)(h * w);

    // 1. Inputs
    auto image_sq = image.square();
    auto input_stack = torch::cat({image, image_sq}, 0);

    // Broadcast kernel
    torch::Tensor kernel_arg;
    int64_t k_el = ones_kernel.numel();
    int64_t k_flat = k_el / (h * w);

    if (k_flat == 1) {
        kernel_arg = ones_kernel;
    } else {
        kernel_arg = torch::cat({ones_kernel, ones_kernel}, 0);
    }

    // 2. FFT
    auto sums = fft_cc_forward_cuda(input_stack, kernel_arg);
    
    // 3. Split
    auto chunks = sums.chunk(2, 0);
    auto s1 = chunks[0];
    auto s2 = chunks[1];

    // 4. Fused Std
    auto out = torch::empty_like(s1);
    int numel = s1.numel();
    int threads = 1024;
    int blocks = (numel + threads - 1) / threads; // Simple grid

    AT_DISPATCH_FLOATING_TYPES(s1.scalar_type(), "local_std_forward_cuda_kernel", ([&] {
        fused_std_kernel<scalar_t><<<blocks, threads>>>(
            s1.data_ptr<scalar_t>(),
            s2.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            static_cast<scalar_t>(N),
            static_cast<scalar_t>(eps),
            numel);
    }));

    // Clone s1 to free sums memory
    return {out, s1.clone()};
}

torch::Tensor local_std_backward_cuda(torch::Tensor grad_output, torch::Tensor image, torch::Tensor ones_kernel, torch::Tensor std_map, torch::Tensor s1, double eps) {
    auto h = ones_kernel.size(-2);
    auto w = ones_kernel.size(-1);
    double N = (double)(h * w);

    // 1. Prepare Grads (dS1, dS2) - Single Allocation
    // grad_stack [2B, ...]
    std::vector<int64_t> stack_shape = s1.sizes().vec();
    stack_shape[0] *= 2;
    auto grad_stack = torch::empty(stack_shape, s1.options());

    auto chunks_g = grad_stack.chunk(2, 0);
    auto grad_s1 = chunks_g[0];
    auto grad_s2 = chunks_g[1];

    int numel = s1.numel();
    int threads_p = 1024;
    int blocks_p = (numel + threads_p - 1) / threads_p;

    AT_DISPATCH_FLOATING_TYPES(s1.scalar_type(), "local_std_backward_prep_kernel", ([&] {
        fused_backward_prep_kernel<scalar_t><<<blocks_p, threads_p>>>(
            grad_output.data_ptr<scalar_t>(),
            std_map.data_ptr<scalar_t>(),
            s1.data_ptr<scalar_t>(),
            grad_s1.data_ptr<scalar_t>(),
            grad_s2.data_ptr<scalar_t>(),
            static_cast<scalar_t>(N),
            static_cast<scalar_t>(eps),
            numel);
    }));

    // 2. Convolution Backward
    // Skip kernel gradient. Use broadcast kernel.
    torch::Tensor kernel_arg;
    int64_t k_el = ones_kernel.numel();
    int64_t k_flat = k_el / (h * w);

    if (k_flat == 1) {
        kernel_arg = ones_kernel;
    } else {
        kernel_arg = torch::cat({ones_kernel, ones_kernel}, 0);
    }

    // Call with output_mask = {true, false} (compute dI, skip dK)
    // pass empty tensor for image input (not needed if dK is false)
    auto d_inputs_vec = fft_cc_backward_cuda(grad_stack, torch::Tensor(), kernel_arg, {true, false});
    auto d_inputs = d_inputs_vec[0];

    // 3. Combine
    auto chunks = d_inputs.chunk(2, 0);
    auto d_i_s1 = chunks[0];
    auto d_i_s2 = chunks[1];
    
    auto d_image = torch::empty_like(image);
    int numel_i = image.numel();
    int blocks_i = (numel_i + threads_p - 1) / threads_p;

    AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "local_std_backward_combine_kernel", ([&] {
        fused_backward_combine_kernel<scalar_t><<<blocks_i, threads_p>>>(
            d_i_s1.data_ptr<scalar_t>(),
            d_i_s2.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            d_image.data_ptr<scalar_t>(),
            numel_i);
    }));

    return d_image;
}
