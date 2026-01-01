#include <torch/extension.h>
#include <vector>
#include <cmath>
#include "local_std.h"
#include "fft_cc.h"

// =========================================================================
// CPU IMPLEMENTATION
// =========================================================================

std::vector<torch::Tensor> local_std_forward_cpu(torch::Tensor image, torch::Tensor ones_kernel, double eps) {
    auto B = image.size(0);
    int64_t H = image.size(-2);
    int64_t W = image.size(-1);

    auto h = ones_kernel.size(-2);
    auto w = ones_kernel.size(-1);
    double N = (double)(h * w);

    // 1. Prepare Inputs
    auto image_sq = image.square();
    auto input_stack = torch::cat({image, image_sq}, 0);

    // Kernel Handling
    // If ones_kernel is (1, 1, h, w), flattened size is 1. We can broadcast it.
    // If ones_kernel is (B, C, h, w), flattened is B*C. We need to duplicate it to match input (2*B*C).

    torch::Tensor kernel_arg;
    int64_t k_el = ones_kernel.numel();
    int64_t k_flat = k_el / (h * w);

    if (k_flat == 1) {
        kernel_arg = ones_kernel;
    } else {
        kernel_arg = torch::cat({ones_kernel, ones_kernel}, 0);
    }

    // 2. Compute Sums via FFT
    auto sums = fft_cc_forward_cpu(input_stack, kernel_arg);
    
    // 3. Split (Views)
    auto chunks = sums.chunk(2, 0);
    auto s1 = chunks[0]; // View
    auto s2 = chunks[1]; // View

    // 4. Compute Std (Fused Loop)
    auto out = torch::empty_like(s1);
    
    // Ensure contiguous for raw pointer access
    auto s1_c = s1.contiguous();
    auto s2_c = s2.contiguous();
    
    int64_t numel = s1_c.numel();

    AT_DISPATCH_FLOATING_TYPES(s1_c.scalar_type(), "local_std_forward_cpu", ([&] {
        const scalar_t* s1_ptr = s1_c.data_ptr<scalar_t>();
        const scalar_t* s2_ptr = s2_c.data_ptr<scalar_t>();
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        scalar_t N_val = static_cast<scalar_t>(N);
        scalar_t eps_val = static_cast<scalar_t>(eps);

        at::parallel_for(0, numel, 2048, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; i++) {
                scalar_t val_s1 = s1_ptr[i];
                scalar_t val_s2 = s2_ptr[i];
                
                scalar_t var = (val_s2 - (val_s1 * val_s1) / N_val) / N_val;
                if (var < 0) var = 0;
                out_ptr[i] = std::sqrt(var + eps_val);
            }
        });
    }));
    
    return {out, s1.clone()};
}

torch::Tensor local_std_backward_cpu(torch::Tensor grad_output, torch::Tensor image, torch::Tensor ones_kernel, torch::Tensor std_map, torch::Tensor s1, double eps) {
    auto h = ones_kernel.size(-2);
    auto w = ones_kernel.size(-1);
    double N = (double)(h * w);
    
    // 1. Prepare Grads Setup
    auto s1_c = s1.contiguous();
    auto std_map_c = std_map.contiguous();
    auto grad_out_c = grad_output.contiguous();

    std::vector<int64_t> stack_shape = s1.sizes().vec();
    stack_shape[0] *= 2;
    auto grad_stack = torch::empty(stack_shape, s1.options());
    
    auto chunks = grad_stack.chunk(2, 0);
    auto grad_s1 = chunks[0];
    auto grad_s2 = chunks[1];

    int64_t numel = s1_c.numel();

    // 2. Compute Grads (dS1, dS2) directly into stack
    AT_DISPATCH_FLOATING_TYPES(s1_c.scalar_type(), "local_std_backward_prep_cpu", ([&] {
        const scalar_t* g_std_ptr = grad_out_c.data_ptr<scalar_t>();
        const scalar_t* std_ptr = std_map_c.data_ptr<scalar_t>();
        const scalar_t* s1_ptr = s1_c.data_ptr<scalar_t>();
        
        scalar_t* ds1_ptr = grad_s1.data_ptr<scalar_t>();
        scalar_t* ds2_ptr = grad_s2.data_ptr<scalar_t>();
        
        scalar_t N_val = static_cast<scalar_t>(N);

        at::parallel_for(0, numel, 2048, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; i++) {
                scalar_t g_std = g_std_ptr[i];
                scalar_t std_val = std_ptr[i];
                scalar_t val_s1 = s1_ptr[i];
                
                scalar_t denom = 2.0 * std_val;
                scalar_t d_var = g_std / (denom + 1e-12);
                
                ds2_ptr[i] = d_var / N_val;
                ds1_ptr[i] = d_var * (-2.0 * val_s1 / (N_val * N_val));
            }
        });
    }));

    // 3. Convolution Backward
    // We only need dI (gradient w.r.t input of forward pass).
    // Kernel handling:
    torch::Tensor kernel_arg;
    int64_t k_el = ones_kernel.numel();
    int64_t k_flat = k_el / (h * w);

    if (k_flat == 1) {
        kernel_arg = ones_kernel;
    } else {
        kernel_arg = torch::cat({ones_kernel, ones_kernel}, 0);
    }

    auto d_inputs_vec = fft_cc_backward_cpu(grad_stack, torch::Tensor(), kernel_arg, {true, false});
    auto d_inputs = d_inputs_vec[0]; // (2B, ...)

    // 4. Combine (dI = dI_s1 + 2*I*dI_s2)
    auto d_image = torch::empty_like(image);
    
    // Views
    auto d_chunks = d_inputs.chunk(2, 0);
    auto d_i_s1 = d_chunks[0].contiguous(); 
    auto d_i_s2 = d_chunks[1].contiguous();
    auto img_c = image.contiguous();
    
    int64_t numel_img = image.numel();

    AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "local_std_backward_combine_cpu", ([&] {
        const scalar_t* g1_ptr = d_i_s1.data_ptr<scalar_t>();
        const scalar_t* g2_ptr = d_i_s2.data_ptr<scalar_t>();
        const scalar_t* img_ptr = img_c.data_ptr<scalar_t>();
        scalar_t* out_ptr = d_image.data_ptr<scalar_t>();

        at::parallel_for(0, numel_img, 2048, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; i++) {
                out_ptr[i] = g1_ptr[i] + 2.0 * img_ptr[i] * g2_ptr[i];
            }
        });
    }));
    
    return d_image;
}
