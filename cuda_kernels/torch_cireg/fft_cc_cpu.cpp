#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <complex>
#include <type_traits>
#include "fft_cc.h"

// ==================================================================================
// FORWARD PASS (CPU)
// ==================================================================================
template <typename scalar_t>
torch::Tensor fft_cc_forward_cpu_impl(torch::Tensor image, torch::Tensor kernel) {
    TORCH_CHECK(image.dim() >= 2, "Image must have at least 2 dimensions");
    TORCH_CHECK(kernel.dim() >= 2, "Kernel must have at least 2 dimensions");

    int64_t H = image.size(-2);
    int64_t W = image.size(-1);
    int64_t h = kernel.size(-2);
    int64_t w = kernel.size(-1);

    // Flatten batch dimensions
    int64_t batch_size = image.numel() / (H * W);
    TORCH_CHECK(kernel.numel() / (h * w) == batch_size,
        "Kernel batch size must match image batch size (after flattening)");

    // Ensure contiguous inputs for raw pointer access
    auto image_contig = image.reshape({batch_size, H, W}).contiguous();
    auto kernel_contig = kernel.reshape({batch_size, h, w}).contiguous();

    // 1. Allocation: [B, 2, H, W]
    auto arena = torch::empty({batch_size, 2, H, W}, image.options());

    // 2. Pack
    // Plane 0: Image
    // Plane 1: Kernel

    scalar_t* arena_ptr = arena.data_ptr<scalar_t>();
    const scalar_t* img_ptr = image_contig.data_ptr<scalar_t>();
    const scalar_t* ker_ptr = kernel_contig.data_ptr<scalar_t>();

    int64_t plane_stride = H * W;
    int64_t batch_stride_arena = 2 * plane_stride;
    int64_t batch_stride_img = H * W;
    int64_t batch_stride_ker = h * w;

    // Use parallel_for to fill both planes
    torch::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; ++b) {
            // Plane 0: Image (Size HxW -> HxW)
            scalar_t* dst_img = arena_ptr + b * batch_stride_arena;
            const scalar_t* src_img = img_ptr + b * batch_stride_img;

            // Image is contiguous HxW, Arena plane is contiguous HxW
            std::memcpy(dst_img, src_img, H * W * sizeof(scalar_t));

            // Plane 1: Kernel (Size hxw -> HxW)
            scalar_t* dst_ker = arena_ptr + b * batch_stride_arena + plane_stride;
            const scalar_t* src_ker = ker_ptr + b * batch_stride_ker;

            for(int64_t r=0; r<H; ++r) {
                scalar_t* dr = dst_ker + r*W;
                if(r < h) {
                    const scalar_t* sr = src_ker + r*w;
                    // Copy valid width
                    std::memcpy(dr, sr, w * sizeof(scalar_t));
                    // Zero padding width
                    if(W > w) std::memset(dr + w, 0, (W-w)*sizeof(scalar_t));
                } else {
                    // Zero padding height
                    std::memset(dr, 0, W*sizeof(scalar_t));
                }
            }
        }
    });

    // 3. Batched FFT
    auto arena_spec = torch::fft::rfft2(arena); // [B, 2, H, W_c]
    int64_t W_c = arena_spec.size(-1);

    // 4. Spectral Math: O = I * conj(K)
    using ComplexT = std::complex<scalar_t>;
    ComplexT* spec_ptr = reinterpret_cast<ComplexT*>(arena_spec.data_ptr());

    int64_t plane_spec_stride = H * W_c;
    int64_t batch_spec_stride = 2 * plane_spec_stride;

    torch::parallel_for(0, batch_size * plane_spec_stride, 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            int64_t b = i / plane_spec_stride;
            int64_t idx = i % plane_spec_stride;

            ComplexT* base = spec_ptr + b * batch_spec_stride + idx;
            ComplexT& I = base[0];            // Plane 0
            ComplexT K = base[plane_spec_stride]; // Plane 1

            // O = I * conj(K)
            // Store in Plane 0
            I = I * std::conj(K);
        }
    });

    // 5. IFFT (We only need Plane 0)
    auto res_spec = arena_spec.select(1, 0); // [B, H, W_c]
    auto res_spatial = torch::fft::irfft2(res_spec, {H, W}); // [B, H, W]

    // 6. Crop
    int64_t out_h = H - h + 1;
    int64_t out_w = W - w + 1;

    auto final_out = res_spatial.slice(1, 0, out_h).slice(2, 0, out_w);

    // Return contiguous clone to free arena memory and ensure output layout
    // Also restore shapes
    return final_out.contiguous().reshape(change_h_w_shapes(image, out_h, out_w));
}

torch::Tensor fft_cc_forward_cpu(torch::Tensor image, torch::Tensor kernel) {
    return AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "fft_cc_forward_cpu", [&] {
        return fft_cc_forward_cpu_impl<scalar_t>(image, kernel);
    });
}

// ==================================================================================
// BACKWARD PASS (CPU)
// ==================================================================================
template <typename scalar_t>
std::vector<torch::Tensor> fft_cc_backward_cpu_impl(
    torch::Tensor grad_output,
    torch::Tensor image,
    torch::Tensor kernel
) {
    TORCH_CHECK(image.dim() >= 2, "Image must have at least 2 dimensions");
    TORCH_CHECK(kernel.dim() >= 2, "Kernel must have at least 2 dimensions");

    int64_t H = image.size(-2);
    int64_t W = image.size(-1);
    int64_t h = kernel.size(-2);
    int64_t w = kernel.size(-1);
    int64_t batch_size = image.numel() / (H * W);
    int64_t out_h = H - h + 1;
    int64_t out_w = W - w + 1;

    // Ensure contiguous inputs
    auto image_contig = image.reshape({batch_size, H, W}).contiguous();
    auto kernel_contig = kernel.reshape({batch_size, h, w}).contiguous();
    auto grad_contig = grad_output.reshape({batch_size, out_h, out_w}).contiguous();

    // 1. Allocation
    // Arena IK: [B, 2, H, W] (Contiguous for efficient 2-plane IFFT)
    // Arena G:  [B, 1, H, W] (Separate, we don't need its IFFT)
    auto arena_IK = torch::empty({batch_size, 2, H, W}, image.options());
    auto arena_G  = torch::empty({batch_size, 1, H, W}, image.options());

    scalar_t* ik_ptr = arena_IK.data_ptr<scalar_t>();
    scalar_t* g_ptr  = arena_G.data_ptr<scalar_t>();

    const scalar_t* img_ptr = image_contig.data_ptr<scalar_t>();
    const scalar_t* ker_ptr = kernel_contig.data_ptr<scalar_t>();
    const scalar_t* grad_ptr = grad_contig.data_ptr<scalar_t>();

    int64_t plane_stride = H * W;
    int64_t batch_ik_stride = 2 * plane_stride;
    int64_t batch_g_stride  = 1 * plane_stride;

    torch::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; ++b) {
            // Pack Image -> IK Plane 0
            scalar_t* dst_i = ik_ptr + b * batch_ik_stride;
            const scalar_t* src_i = img_ptr + b * plane_stride;
            std::memcpy(dst_i, src_i, plane_stride * sizeof(scalar_t));

            // Pack Kernel -> IK Plane 1
            scalar_t* dst_k = ik_ptr + b * batch_ik_stride + plane_stride;
            const scalar_t* src_k = ker_ptr + b * h * w;
            for(int64_t r=0; r<H; ++r) {
                scalar_t* dr = dst_k + r*W;
                if(r < h) {
                    const scalar_t* sr = src_k + r*w;
                    std::memcpy(dr, sr, w * sizeof(scalar_t));
                    if(W > w) std::memset(dr + w, 0, (W-w)*sizeof(scalar_t));
                } else {
                    std::memset(dr, 0, W*sizeof(scalar_t));
                }
            }

            // Pack Grad -> G Plane 0
            scalar_t* dst_g = g_ptr + b * batch_g_stride;
            const scalar_t* src_g = grad_ptr + b * out_h * out_w;
            for(int64_t r=0; r<H; ++r) {
                scalar_t* dr = dst_g + r*W;
                if(r < out_h) {
                    const scalar_t* sr = src_g + r*out_w;
                    std::memcpy(dr, sr, out_w * sizeof(scalar_t));
                    if(W > out_w) std::memset(dr + out_w, 0, (W-out_w)*sizeof(scalar_t));
                } else {
                    std::memset(dr, 0, W*sizeof(scalar_t));
                }
            }
        }
    });

    // 2. FFTs
    // IK
    auto spec_IK = torch::fft::rfft2(arena_IK); // [B, 2, H, W_c]
    // G
    auto spec_G  = torch::fft::rfft2(arena_G);  // [B, 1, H, W_c]

    int64_t W_c = spec_IK.size(-1);

    // 3. Spectral Math
    using ComplexT = std::complex<scalar_t>;
    ComplexT* ptr_ik = reinterpret_cast<ComplexT*>(spec_IK.data_ptr());
    ComplexT* ptr_g  = reinterpret_cast<ComplexT*>(spec_G.data_ptr());

    int64_t plane_spec = H * W_c;
    int64_t batch_spec_ik = 2 * plane_spec;
    int64_t batch_spec_g  = 1 * plane_spec;

    torch::parallel_for(0, batch_size * plane_spec, 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            int64_t b = i / plane_spec;
            int64_t idx = i % plane_spec;

            ComplexT* base_ik = ptr_ik + b * batch_spec_ik + idx;
            ComplexT* base_g  = ptr_g  + b * batch_spec_g  + idx;

            ComplexT I = base_ik[0];
            ComplexT K = base_ik[plane_spec];
            ComplexT G = base_g[0];

            // dI = G * K
            base_ik[0] = G * K;

            // dK = I * conj(G)
            base_ik[plane_spec] = I * std::conj(G);
        }
    });

    // 4. IFFT on IK (Contiguous)
    auto out_spatial = torch::fft::irfft2(spec_IK, {H, W}); // [B, 2, H, W]

    // 5. Finalize
    auto dI = out_spatial.select(1, 0); // View [B, H, W]
    auto dT = out_spatial.select(1, 1).slice(1, 0, h).slice(2, 0, w); // View [B, h, w]

    return {
        dI.contiguous().reshape(image.sizes()),
        dT.contiguous().reshape(kernel.sizes())
    };
}

std::vector<torch::Tensor> fft_cc_backward_cpu(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel) {
    return AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "fft_cc_backward_cpu", [&] {
        return fft_cc_backward_cpu_impl<scalar_t>(grad_output, image, kernel);
    });
}
