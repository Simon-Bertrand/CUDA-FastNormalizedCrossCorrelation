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
    int64_t kernel_batch_size = kernel.numel() / (h * w);

    // Support Broadcasting: kernel_batch_size must be 1 or equal to batch_size
    TORCH_CHECK(kernel_batch_size == 1 || kernel_batch_size == batch_size,
        "Kernel batch size must match image batch size or be 1 (broadcasting)");

    bool broadcast_kernel = (kernel_batch_size == 1 && batch_size > 1);

    // Ensure contiguous inputs for raw pointer access
    auto image_contig = image.reshape({batch_size, H, W}).contiguous();
    auto kernel_contig = kernel.reshape({kernel_batch_size, h, w}).contiguous();

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
    int64_t batch_stride_ker = broadcast_kernel ? 0 : (h * w); // If broadcast, stride is 0 (reuse first kernel)

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
            const scalar_t* src_ker = ker_ptr + b * batch_stride_ker; // if broadcast, b * 0 = 0

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
    torch::Tensor kernel,
    std::vector<bool> output_mask
) {
    TORCH_CHECK(grad_output.dim() >= 2, "Grad must be >= 2D");

    // Need image dims for full size
    // Need kernel dims for crop size
    // Note: If image is dummy (e.g. not computing dK), we assume its size matches grad (adjusted for correlation).
    // Actually we need 'H, W' which is defined by image size.
    // If output_mask[1] (dK) is false, 'image' might be undefined/dummy?
    // But we need H, W.
    // Assuming image is always provided or at least its sizes are valid.
    // The previous implementation used image.size(-2).

    // If compute_dK is false, we don't need 'image' data, but we need 'H, W'.
    // If compute_dI is false, we don't need 'kernel' data, but we need 'h, w'.

    bool compute_dI = output_mask[0];
    bool compute_dK = output_mask[1];

    int64_t h = kernel.size(-2);
    int64_t w = kernel.size(-1);
    int64_t H, W;

    if (image.defined()) {
        H = image.size(-2);
        W = image.size(-1);
    } else {
        int64_t out_h = grad_output.size(-2);
        int64_t out_w = grad_output.size(-1);
        H = out_h + h - 1;
        W = out_w + w - 1;
    }

    int64_t out_h = H - h + 1;
    int64_t out_w = W - w + 1;

    int64_t batch_size = grad_output.numel() / (out_h * out_w);
    int64_t kernel_batch_size = kernel.numel() / (h * w);

    bool broadcast_kernel = (kernel_batch_size == 1 && batch_size > 1);

    auto grad_contig = grad_output.reshape({batch_size, out_h, out_w}).contiguous();

    // Optimizations:
    // If only dI needed: Need G and K. Compute G*K.
    // If only dK needed: Need I and G. Compute I*conj(G).
    // If both needed: Need I, K, G.

    // We will use a flexible arena strategy.
    // Always need G (Grad).

    // Arena Plan:
    // Plane 0: G (always needed)
    // Plane 1: K (needed for dI) or I (needed for dK)
    // Plane 2: I (needed for dK if both)

    // Actually, simpler logic:
    // Case 1: Both (3 planes: I, K, G). Reuse existing logic.
    // Case 2: dI only (2 planes: G, K). dI = G * K.
    // Case 3: dK only (2 planes: I, G). dK = I * conj(G).

    int64_t num_planes = 0;
    if (compute_dI && compute_dK) num_planes = 3;
    else if (compute_dI || compute_dK) num_planes = 2;
    else return {torch::Tensor(), torch::Tensor()};

    auto arena = torch::empty({batch_size, num_planes, H, W}, grad_output.options());

    // Data Pointers
    const scalar_t* img_ptr = nullptr;
    const scalar_t* ker_ptr = nullptr;
    const scalar_t* grad_ptr = grad_contig.data_ptr<scalar_t>();

    torch::Tensor image_contig;
    if (compute_dK) {
        image_contig = image.contiguous();
        img_ptr = image_contig.data_ptr<scalar_t>();
    }
    if (compute_dI) ker_ptr = kernel.contiguous().data_ptr<scalar_t>(); // broadcast handled by stride

    scalar_t* arena_ptr = arena.data_ptr<scalar_t>();

    int64_t plane_stride = H * W;
    int64_t batch_stride_arena = num_planes * plane_stride;

    int64_t batch_stride_ker = broadcast_kernel ? 0 : (h * w);
    int64_t batch_stride_img = H * W;

    // PACKING
    torch::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; ++b) {
            scalar_t* base = arena_ptr + b * batch_stride_arena;

            if (compute_dI && compute_dK) {
                // Plane 0: I
                // Plane 1: K
                // Plane 2: G
                scalar_t* dst_i = base;
                scalar_t* dst_k = base + plane_stride;
                scalar_t* dst_g = base + 2*plane_stride;

                // Pack I
                std::memcpy(dst_i, img_ptr + b * batch_stride_img, plane_stride * sizeof(scalar_t));

                // Pack K
                const scalar_t* src_k = ker_ptr + b * batch_stride_ker;
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

                // Pack G
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

            } else if (compute_dI) {
                // Only dI -> Need G and K
                // Plane 0: G
                // Plane 1: K
                scalar_t* dst_g = base;
                scalar_t* dst_k = base + plane_stride;

                // Pack G
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

                // Pack K
                const scalar_t* src_k = ker_ptr + b * batch_stride_ker;
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

            } else {
                // Only dK -> Need I and G
                // Plane 0: I
                // Plane 1: G
                scalar_t* dst_i = base;
                scalar_t* dst_g = base + plane_stride;

                // Pack I
                std::memcpy(dst_i, img_ptr + b * batch_stride_img, plane_stride * sizeof(scalar_t));

                // Pack G
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
        }
    });

    // FFT
    auto spec = torch::fft::rfft2(arena); // [B, num_planes, H, W_c]
    int64_t W_c = spec.size(-1);

    // Spectral Math
    using ComplexT = std::complex<scalar_t>;
    ComplexT* spec_ptr = reinterpret_cast<ComplexT*>(spec.data_ptr());
    int64_t plane_spec = H * W_c;
    int64_t batch_spec = num_planes * plane_spec;

    torch::parallel_for(0, batch_size * plane_spec, 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            int64_t b = i / plane_spec;
            int64_t idx = i % plane_spec;
            ComplexT* base = spec_ptr + b * batch_spec + idx;

            if (compute_dI && compute_dK) {
                // 0: I, 1: K, 2: G
                ComplexT I = base[0];
                ComplexT K = base[plane_spec];
                ComplexT G = base[2*plane_spec];

                // dI = G * K (Store in 0)
                base[0] = G * K;

                // dK = I * conj(G) (Store in 1)
                base[plane_spec] = I * std::conj(G);

            } else if (compute_dI) {
                // 0: G, 1: K
                ComplexT G = base[0];
                ComplexT K = base[plane_spec];
                // dI = G * K (Store in 0)
                base[0] = G * K;
            } else {
                // 0: I, 1: G
                ComplexT I = base[0];
                ComplexT G = base[plane_spec];
                // dK = I * conj(G) (Store in 1) (Wait, dK should be output. We can reuse 0 or 1)
                // Let's store in 0 for simplicity of IFFT slicing?
                // Or store in 1?
                // Let's store in 0.
                base[0] = I * std::conj(G);
            }
        }
    });

    // IFFT
    // We only need specific planes.
    // If Both: Need Plane 0 (dI) and Plane 1 (dK). Plane 2 is garbage.
    // If Single: Need Plane 0 (Result). Plane 1 is garbage.

    int64_t ifft_planes = (compute_dI && compute_dK) ? 2 : 1;
    auto spec_sliced = spec.slice(1, 0, ifft_planes);
    auto spatial = torch::fft::irfft2(spec_sliced, {H, W}); // [B, ifft_planes, H, W]

    torch::Tensor dI, dK;

    if (compute_dI && compute_dK) {
        dI = spatial.select(1, 0).contiguous().reshape(image.sizes());
        auto dT = spatial.select(1, 1).slice(1, 0, h).slice(2, 0, w);
        if (broadcast_kernel) {
            dK = dT.sum(0, true).reshape(kernel.sizes());
        } else {
             dK = dT.contiguous().reshape(kernel.sizes());
        }

    } else if (compute_dI) {
        if (image.defined()) {
            dI = spatial.select(1, 0).contiguous().reshape(image.sizes());
        } else {
            // Reconstruct shape from H, W and batch
            // Assuming image was (B, C, H, W) flattened.
            // We don't know C from here easily if only B is known.
            // But we treat it as flattened B.
            // Return (B, H, W).
            dI = spatial.select(1, 0).contiguous().reshape({batch_size, H, W});
        }
    } else {
        auto dT = spatial.select(1, 0).slice(1, 0, h).slice(2, 0, w);
        if (broadcast_kernel) {
             dK = dT.sum(0, true).reshape(kernel.sizes());
        } else {
             dK = dT.contiguous().reshape(kernel.sizes());
        }
    }

    return {dI, dK};
}

std::vector<torch::Tensor> fft_cc_backward_cpu(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel, std::vector<bool> output_mask) {
    return AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "fft_cc_backward_cpu", [&] {
        return fft_cc_backward_cpu_impl<scalar_t>(grad_output, image, kernel, output_mask);
    });
}
