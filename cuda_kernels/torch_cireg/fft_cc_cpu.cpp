#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "fft_cc.h"

// ==================================================================================
// FORWARD PASS (CPU)
// ==================================================================================
template <typename scalar_t>
torch::Tensor fft_cc_forward_cpu_impl(torch::Tensor image, torch::Tensor kernel) {
    // 1. Checks and Flattening
    TORCH_CHECK(image.dim() >= 2, "Image must have at least 2 dimensions");
    TORCH_CHECK(kernel.dim() >= 2, "Kernel must have at least 2 dimensions");

    auto img_sizes = image.sizes();
    int64_t H = img_sizes[img_sizes.size() - 2];
    int64_t W = img_sizes[img_sizes.size() - 1];
    int64_t h = kernel.size(-2);
    int64_t w = kernel.size(-1);

    int64_t batch_size = image.numel() / (H * W);
    TORCH_CHECK(kernel.numel() / (h * w) == batch_size,
        "Kernel batch size must match image batch size (after flattening)");

    auto img_flat = image.reshape({batch_size, H, W});
    auto ker_flat = kernel.reshape({batch_size, h, w});

    // 2. Setup Arena: [B, 2, H, W]
    // Plane 0: Image
    // Plane 1: Kernel (Padded)
    auto arena = torch::zeros({batch_size, 2, H, W}, image.options());

    using namespace torch::indexing;
    // Copy Image to Plane 0
    arena.select(1, 0).copy_(img_flat);
    // Copy Kernel to Plane 1 (Top-Left aligned for FFT)
    arena.select(1, 1).index_put_({Slice(), Slice(0, h), Slice(0, w)}, ker_flat);

    // 3. Batched FFT
    auto arena_flat = arena.reshape({batch_size * 2, H, W});
    auto arena_spec = torch::fft::rfft2(arena_flat);

    int64_t W_c = arena_spec.size(-1);
    auto spec_view = arena_spec.reshape({batch_size, 2, H, W_c});

    // 4. Spectral Math: O = I * conj(K)
    // We store result in Plane 0
    auto I_spec = spec_view.select(1, 0);
    auto K_spec = spec_view.select(1, 1);

    I_spec.mul_(K_spec.conj());

    // 5. IFFT
    auto res_spec = I_spec; // Plane 0
    auto res_spatial = torch::fft::irfft2(res_spec, {H, W}); // [B, H, W]

    // 6. Crop
    int64_t out_h = H - h + 1;
    int64_t out_w = W - w + 1;

    auto final_out = res_spatial.slice(1, 0, out_h).slice(2, 0, out_w);

    return final_out.reshape(change_h_w_shapes(image, out_h, out_w));
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
    // 1. Checks
    auto img_sizes = image.sizes();
    int64_t H = img_sizes[img_sizes.size() - 2];
    int64_t W = img_sizes[img_sizes.size() - 1];
    int64_t h = kernel.size(-2);
    int64_t w = kernel.size(-1);
    int64_t batch_size = image.numel() / (H * W);

    // Expected Output Size
    int64_t out_h = H - h + 1;
    int64_t out_w = W - w + 1;

    auto grad_flat = grad_output.reshape({batch_size, out_h, out_w});
    auto img_flat = image.reshape({batch_size, H, W});
    auto ker_flat = kernel.reshape({batch_size, h, w});

    // 2. Setup Arena: [B, 3, H, W]
    // Plane 0: Image
    // Plane 1: Kernel (Padded)
    // Plane 2: GradOutput (Padded)
    auto arena = torch::zeros({batch_size, 3, H, W}, image.options());
    using namespace torch::indexing;

    arena.select(1, 0).copy_(img_flat);
    arena.select(1, 1).index_put_({Slice(), Slice(0, h), Slice(0, w)}, ker_flat);
    arena.select(1, 2).index_put_({Slice(), Slice(0, out_h), Slice(0, out_w)}, grad_flat);

    // 3. Batched FFT
    auto arena_flat = arena.reshape({batch_size * 3, H, W});
    auto arena_spec = torch::fft::rfft2(arena_flat);
    int64_t W_c = arena_spec.size(-1);
    auto spec_view = arena_spec.reshape({batch_size, 3, H, W_c});

    // 4. Spectral Math
    // Plane 0: I_spec
    // Plane 1: K_spec
    // Plane 2: G_spec (GradOutput)

    // Grad Image (dI) = G * K
    // Grad Kernel (dK) = I * conj(G)

    // Reuse buffers:
    // Store dI in Plane 0
    // Store dK in Plane 1

    // Careful with dependencies: I used in dK, K used in dI, G used in both.

    // Clone necessary inputs if overwriting.
    // Or just compute into temp and write back.

    // spec_view is a view, data is contiguous in arena_spec? No, strides.
    // But we can iterate.

    auto I_s = spec_view.select(1, 0);
    auto K_s = spec_view.select(1, 1);
    auto G_s = spec_view.select(1, 2);

    // We can't overwrite I or K yet.
    // Actually we have 3 planes. We need 2 outputs.
    // We can overwrite P0 with dI, P1 with dK.
    // dI depends on G and K.
    // dK depends on I and G.

    // Temp buffer for dI or dK?
    // Let's compute dI = G * K. Overwrite P0? No, P0 is I, needed for dK.
    // Let's compute dK = I * conj(G). Overwrite P1? Yes, K is not needed for dK.
    // BUT K is needed for dI.

    // So we need to compute dI first? dI = G * K. P0 = P2 * P1. Overwrites I.
    // Then dK = I * conj(G). But I is gone.

    // Solution: Clone I or K or use a temp tensor.
    // Or just allocate output tensors separately? No, want to use batched IFFT.

    // Simple way:
    // dI_spec = G * K
    // dK_spec = I * conj(G)
    // Write dI_spec to P0, dK_spec to P1.
    // Do it in a loop or using temps.

    auto dI_spec = G_s * K_s; // New allocation (small, just freq domain)

    // Now compute dK into P1 (overwriting K)
    // dK = I * conj(G)
    K_s.copy_(I_s);
    K_s.mul_(G_s.conj());

    // Now copy dI into P0 (overwriting I)
    I_s.copy_(dI_spec);

    // 5. IFFT (Only need first 2 planes)
    // spec_view slice 0..2
    auto out_spec = spec_view.slice(1, 0, 2).reshape({batch_size * 2, H, W_c});
    auto out_spatial = torch::fft::irfft2(out_spec, {H, W}).reshape({batch_size, 2, H, W});

    // 6. Finalize
    auto dI = out_spatial.select(1, 0); // Full HxW
    auto dT_full = out_spatial.select(1, 1);
    auto dT = dT_full.slice(1, 0, h).slice(2, 0, w); // Crop to hxw

    // Reshape to original shapes
    auto dI_out = dI.reshape(image.sizes());
    auto dT_out = dT.reshape(kernel.sizes());

    return {dI_out, dT_out};
}

std::vector<torch::Tensor> fft_cc_backward_cpu(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel) {
    return AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "fft_cc_backward_cpu", [&] {
        return fft_cc_backward_cpu_impl<scalar_t>(grad_output, image, kernel);
    });
}
