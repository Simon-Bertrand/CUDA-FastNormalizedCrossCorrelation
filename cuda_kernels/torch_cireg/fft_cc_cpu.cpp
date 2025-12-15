#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <iostream>

// Helper: Reshape Output
std::vector<int64_t> change_h_w_shapes(const torch::Tensor& tensor, const int64_t& H, const int64_t& W) {
    std::vector<int64_t> out;
    out.reserve(tensor.dim());
    out.insert(out.end(), tensor.sizes().begin(), tensor.sizes().end() - 2);
    out.insert(out.end(), {H,W});
    return out;
}

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
    // Use select().copy_() which is generally cleaner/faster than generic index_put_
    arena.select(1, 0).copy_(img_flat);

    // Fill Plane 1: Image^2 (ZNCC) or Kernel (CC)
    // Fill Plane 2: Kernel (ZNCC)
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

        // Plane 1: Image^2
        // Calculate square directly into the arena to avoid temporary allocation
        auto plane1 = arena.select(1, 1);
        torch::mul_out(plane1, img_flat, img_flat);

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

    // 4. Spectral Math
    // Compute products in-place where possible, or reuse spec buffer for IFFT inputs.
    // We need to form the inputs for IFFT.
    // ZNCC Outputs: Num, Sum_I, Sum_I2
    // CC Output: Num

    // We can reuse the same arena_spec buffer for IFFT inputs.
    // Let's overwrite:
    // Plane 0 -> Result/Num
    // Plane 1 -> Sum_I (ZNCC)
    // Plane 2 -> Sum_I2 (ZNCC)

    // Accessors
    auto I_spec = spec_view.select(1, 0);

    if (normalize) {
        auto I2_spec   = spec_view.select(1, 1);
        auto K_spec    = spec_view.select(1, 2);
        auto Ones_spec = spec_view.select(1, 3);

        // Be careful with aliasing if we write back to I_spec.
        // We need:
        // Out0 = I * conj(K)
        // Out1 = I * conj(Ones)
        // Out2 = I2 * conj(Ones)

        // We can compute into temporary or carefully order.
        // If we write to Out0 (I_spec), we lose I_spec for Out1.
        // So we need a temporary for I_spec or compute Out1 first if it doesn't overlap?
        // Actually, we can just allocate a new tensor for the spectral product results?
        // But we want to minimize allocation.
        // Better to reuse planes 0, 1, 2.

        // Copy I_spec if needed.
        auto I_spec_clone = I_spec.clone();

        // We must be careful with the order of operations to avoid overwriting inputs
        // before they are used, since we are writing back to the same arena planes.
        // Planes: 0=I, 1=I2, 2=K, 3=Ones
        // Outputs needed:
        // Out0 (Num)    = I * conj(K)    -> Write to Plane 0
        // Out1 (Sum_I)  = I * conj(Ones) -> Write to Plane 1
        // Out2 (Sum_I2) = I2 * conj(Ones)-> Write to Plane 2

        // 1. Compute Out0 first. Needs K (Plane 2). Writes to Plane 0 (Overwrites I).
        //    We use I_spec_clone, so overwriting I is fine.
        spec_view.select(1, 0).copy_(I_spec_clone * K_spec.conj());

        // 2. Compute Out2. Needs I2 (Plane 1). Writes to Plane 2 (Overwrites K).
        //    K is no longer needed (used in step 1).
        spec_view.select(1, 2).copy_(I2_spec * Ones_spec.conj());

        // 3. Compute Out1. Needs Ones (Plane 3). Writes to Plane 1 (Overwrites I2).
        //    I2 is no longer needed (used in step 2).
        spec_view.select(1, 1).copy_(I_spec_clone * Ones_spec.conj());

        // We don't need Plane 3 anymore.

    } else {
        auto K_spec = spec_view.select(1, 1);
        // Out0 = I * conj(K) -> Write to Plane 0
        spec_view.select(1, 0).mul_(K_spec.conj());
    }

    // 5. Batched IFFT
    // We only need to transform the relevant planes.
    // ZNCC: 3 planes (0, 1, 2).
    // CC: 1 plane (0).

    int result_depth = normalize ? 3 : 1;

    // Slice the relevant part of the flattened spec
    // Note: The reshaping [B*D, ...] interleaves the batch and depth.
    // spec_view is [B, D, H, Wc].
    // We want to IFFT [B, result_depth, H, Wc].

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
        auto mask = (var_term < static_cast<scalar_t>(1e-5)) | (ker_norms.view({batch_size, 1, 1}) < static_cast<scalar_t>(1e-6));

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
