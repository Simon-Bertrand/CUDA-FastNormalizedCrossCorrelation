#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "fft_cc.h"

// Note: change_h_w_shapes is now inline in fft_cc.h

template <typename scalar_t>
torch::Tensor fft_cross_correlation_cpu_impl(
    torch::Tensor image,
    torch::Tensor kernel,
    bool normalize
) {
    // 1. Checks and Flattening
    TORCH_CHECK(image.dim() >= 2, "Image must have at least 2 dimensions");
    TORCH_CHECK(kernel.dim() >= 2, "Kernel must have at least 2 dimensions");

    // We treat all dims except last 2 as batch.
    auto img_sizes = image.sizes();
    int64_t H = img_sizes[img_sizes.size() - 2];
    int64_t W = img_sizes[img_sizes.size() - 1];
    int64_t h = kernel.size(-2);
    int64_t w = kernel.size(-1);

    // Flatten batch dims
    int64_t batch_size = image.numel() / (H * W);

    // Ensure kernel batch matches
    TORCH_CHECK(kernel.numel() / (h * w) == batch_size,
        "Kernel batch size must match image batch size (after flattening)");

    auto img_flat = image.reshape({batch_size, H, W});
    auto ker_flat = kernel.reshape({batch_size, h, w});

    // 2. Setup Optimization "Arena" (Stacked Tensors)
    // To minimize RFFT calls, we stack all needed planes into one large batch.
    // ZNCC needs: I, I^2, K, Ones -> 4 planes
    // CC needs:   I, K           -> 2 planes

    int depth = normalize ? 4 : 2;
    int64_t total_batch = batch_size * depth;

    // Allocate arena: [B, Depth, H, W] -> View as [B*Depth, H, W]
    auto arena = torch::zeros({batch_size, depth, H, W}, image.options());

    using namespace torch::indexing;

    // Fill Plane 0: Image
    // Optimized: In ZNCC, subtract per-image mean to improve numerical stability.

    if (normalize) {
        // Calculate Mean per image: [B, 1, 1]
        auto img_means = img_flat.mean({1, 2}, true);

        // Plane 0: Image (Centered)
        // Copy and subtract mean in-place
        auto plane0 = arena.select(1, 0);
        plane0.copy_(img_flat);
        plane0.sub_(img_means);

        // Plane 1: Image^2 (ZNCC)
        // Calculate square directly into the arena from the centered image
        auto plane1 = arena.select(1, 1);
        torch::mul_out(plane1, plane0, plane0);
    } else {
        // Just copy image
        arena.select(1, 0).copy_(img_flat);
    }

    // Fill Plane 2: Kernel (ZNCC) or Plane 1 (CC)
    // Fill Plane 3: Ones (ZNCC)

    torch::Tensor ker_norms;
    scalar_t sqrt_N = static_cast<scalar_t>(1.0);

    // Prepare Kernel Data
    torch::Tensor ker_to_pack;
    if (normalize) {
        // Pre-process Kernel
        auto k_mean = ker_flat.mean({1, 2}, true);
        auto k_centered = ker_flat - k_mean;
        ker_norms = k_centered.norm(2, {1, 2});
        ker_to_pack = k_centered;
        sqrt_N = std::sqrt(static_cast<scalar_t>(h * w));

        // Plane 2: Kernel (Padded)
        arena.select(1, 2).index_put_({Slice(), Slice(0, h), Slice(0, w)}, ker_to_pack);

        // Plane 3: Ones (Padded)
        arena.select(1, 3).index_put_({Slice(), Slice(0, h), Slice(0, w)}, static_cast<scalar_t>(1.0));

    } else {
        // Plane 1: Kernel (Padded)
        arena.select(1, 1).index_put_({Slice(), Slice(0, h), Slice(0, w)}, ker_flat);
    }

    // 3. Batched FFT
    // Input: [B, D, H, W] -> Reshape to [B*D, H, W]
    auto arena_flat = arena.reshape({total_batch, H, W});
    auto arena_spec = torch::fft::rfft2(arena_flat);

    // View back to [B, D, H, W_complex]
    int64_t W_complex = arena_spec.size(-1);
    auto spec_view = arena_spec.reshape({batch_size, depth, H, W_complex});

    // 4. Spectral Math (Optimized In-Place)
    // We reuse the same arena_spec buffer for IFFT inputs.
    // ZNCC Target Layout for IFFT: Plane 0 -> Num, Plane 1 -> Sum_I, Plane 2 -> Sum_I2.

    if (normalize) {
        // Initial Layout: 0=I, 1=I2, 2=K, 3=Ones
        // Accessors via indices to allow in-place modification

        // 1. Compute Sum_I2 (needs I2, Ones). Store in Plane 1.
        // P1 = P1 * conj(P3). Replaces I2.
        spec_view.select(1, 1).mul_(spec_view.select(1, 3).conj());

        // 2. Compute Sum_I (needs I, Ones). Store in Plane 3.
        // P3 = P0 * conj(P3). Replaces Ones.
        // Note: P3 was used in step 1, but we didn't modify it there (only P1).
        // Now we modify P3.
        // We need conj(Ones). P3 is Ones.
        // Fixed: Use copy_(conj()) since conj_() is not available
        {
            auto p3 = spec_view.select(1, 3);
            p3.copy_(p3.conj()); // In-place conjugate physically
            p3.mul_(spec_view.select(1, 0)); // P3 = conj(Ones) * I
        }

        // 3. Compute Num (needs I, K). Store in Plane 0.
        // P0 = P0 * conj(P2). Replaces I.
        spec_view.select(1, 0).mul_(spec_view.select(1, 2).conj());

        // Current Layout: 0=Num, 1=Sum_I2, 2=K, 3=Sum_I
        // Target Layout:  0=Num, 1=Sum_I,  2=Sum_I2

        // 4. Shuffle
        // Move Sum_I2 (from P1) to P2 (overwrite K)
        spec_view.select(1, 2).copy_(spec_view.select(1, 1));

        // Move Sum_I (from P3) to P1 (overwrite old Sum_I2)
        spec_view.select(1, 1).copy_(spec_view.select(1, 3));

        // Result: 0=Num, 1=Sum_I, 2=Sum_I2. Correct.
    } else {
        auto K_spec = spec_view.select(1, 1);
        // Out0 = I * conj(K) -> Write to Plane 0
        spec_view.select(1, 0).mul_(K_spec.conj());
    }

    // 5. Batched IFFT
    int result_depth = normalize ? 3 : 1;
    auto result_spec = spec_view.slice(1, 0, result_depth).reshape({batch_size * result_depth, H, W_complex});
    auto result_spatial_flat = torch::fft::irfft2(result_spec, {H, W});
    auto result_spatial = result_spatial_flat.reshape({batch_size, result_depth, H, W});

    // 6. Finalize & Crop
    torch::Tensor final_out;
    int64_t crop_h = H - h + 1;
    int64_t crop_w = W - w + 1;

    if (normalize) {
        auto num    = result_spatial.select(1, 0);
        auto sum_i  = result_spatial.select(1, 1);
        auto sum_i2 = result_spatial.select(1, 2);

        scalar_t N = static_cast<scalar_t>(h * w);
        auto var_term = N * sum_i2 - sum_i.square();
        var_term = torch::relu(var_term);

        auto inv_std = torch::rsqrt(var_term);

        scalar_t epsilon = std::is_same<scalar_t, double>::value ? static_cast<scalar_t>(1e-6) : static_cast<scalar_t>(1e-5);

        auto mask = (var_term < epsilon) | (ker_norms.view({batch_size, 1, 1}) < epsilon);

        final_out = (num * sqrt_N) * inv_std * (static_cast<scalar_t>(1.0) / ker_norms.view({batch_size, 1, 1}));
        final_out.index_put_({mask}, static_cast<scalar_t>(0.0));
    } else {
        final_out = result_spatial.select(1, 0);
    }

    final_out = final_out.slice(1, 0, crop_h).slice(2, 0, crop_w);

    return final_out.reshape(change_h_w_shapes(image, crop_h, crop_w));
}

torch::Tensor fft_cross_correlation_cpu(
    torch::Tensor image,
    torch::Tensor kernel,
    bool normalize = false
) {
    return AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "fft_cross_correlation_cpu", [&] {
        return fft_cross_correlation_cpu_impl<scalar_t>(image, kernel, normalize);
    });
}
