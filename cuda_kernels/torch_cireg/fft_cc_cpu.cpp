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

torch::Tensor fft_cross_correlation_cpu(
    torch::Tensor image,
    torch::Tensor kernel,
    bool normalize = false
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

    // 2. Pre-process Kernel (Normalization)
    torch::Tensor ker_processed = ker_flat;
    torch::Tensor ker_norms;
    float sqrt_N = 1.0f;

    if (normalize) {
        // Compute mean and center
        // Kernel shape: [B, h, w]
        auto k_mean = ker_flat.mean({1, 2}, true); // [B, 1, 1]
        auto k_centered = ker_flat - k_mean;

        // Compute norm
        ker_norms = k_centered.norm(2, {1, 2}); // [B]

        ker_processed = k_centered;
        sqrt_N = std::sqrt((float)(h * w));
    }

    // 3. Pad Kernel to Image Size
    // We need to pad kernel to (H, W).
    // The existing CUDA code seems to do implicit padding/packing during copy.
    // Here we can use F.pad or create a new tensor.
    // To match correlation, we need to be careful about alignment.
    // FFT correlation: I * conj(K).
    // Usually K is padded with zeros.

    // Create packed kernel tensor: [B, H, W]
    auto ker_padded = torch::zeros({batch_size, H, W}, image.options());

    // Copy kernel into the top-left corner (or wherever appropriate for standard correlation)
    // The CUDA code packs it:
    // int k_x = idx % kernel_w;
    // int k_y = idx / kernel_w;
    // arena_ptr[batch_base] = k_val;
    // So it copies to top-left (0,0).

    using namespace torch::indexing;
    ker_padded.index_put_({Slice(), Slice(0, h), Slice(0, w)}, ker_processed);

    // 4. FFTs
    // rfft2 expects real input, returns complex.
    // We use signal_ndim=2 to transform last 2 dims.
    auto img_fft = torch::fft::rfft2(img_flat);
    auto ker_fft = torch::fft::rfft2(ker_padded);

    // 5. Spectral Math
    // ZNCC needs:
    // Term 1: num = IFFT( FFT(I) * conj(FFT(T_norm)) )  where T_norm is centered kernel
    // Term 2: sum_i  = IFFT( FFT(I) * conj(FFT(1_window)) ) -> This is local sum of image
    // Term 3: sum_i2 = IFFT( FFT(I^2) * conj(FFT(1_window)) ) -> This is local sum of squared image

    // Standard CC needs:
    // result = IFFT( FFT(I) * conj(FFT(T)) )

    torch::Tensor result;

    if (normalize) {
        // Create the "1_window" kernel (ones where kernel is defined)
        auto ones_kernel = torch::zeros({batch_size, H, W}, image.options());
        ones_kernel.index_put_({Slice(), Slice(0, h), Slice(0, w)}, 1.0f);
        auto ones_fft = torch::fft::rfft2(ones_kernel);

        // Term 1: Numerator (Correlation with centered kernel)
        auto num_spec = img_fft * ker_fft.conj();
        auto num = torch::fft::irfft2(num_spec, {H, W});

        // Term 2 & 3: Local Stats of Image
        // I^2
        auto img2_flat = img_flat.square();
        auto img2_fft = torch::fft::rfft2(img2_flat);

        // Convolve I and I^2 with ones_window
        // To do convolution via correlation formula (which we are effectively doing with conj),
        // we use the same ones_fft.
        // Wait, correlation vs convolution.
        // Cross-Corr(f, g) = f * conj(g).
        // If g is a box filter (symmetric), correlation == convolution.
        // The ones kernel is not symmetric around 0,0 (it's in top-left).
        // But the math in CUDA kernel:
        // cmul_conj_scale(I, O) where O is ones_fft.
        // So it computes Correlation(I, Ones).
        // Which effectively sums pixels under the window sliding.

        auto sum_i_spec = img_fft * ones_fft.conj();
        auto sum_i = torch::fft::irfft2(sum_i_spec, {H, W});

        auto sum_i2_spec = img2_fft * ones_fft.conj();
        auto sum_i2 = torch::fft::irfft2(sum_i2_spec, {H, W});

        // Compute ZNCC score
        // var = N * sum_i2 - sum_i^2
        // score = (num * sqrt_N) / sqrt(var)

        // Avoid negative var due to precision
        float N = (float)(h * w); // Note: CUDA uses h*w as N.

        auto var_term = N * sum_i2 - sum_i.square();
        var_term = torch::relu(var_term); // fmaxf(..., 0.0f)

        // Handle division by zero
        // if var_term < 1e-5 or t_norm < 1e-6 -> 0

        auto inv_std = torch::rsqrt(var_term);
        // Mask out bad values
        auto mask = (var_term < 1e-5) | (ker_norms.view({batch_size, 1, 1}) < 1e-6);

        result = (num * sqrt_N) * inv_std * (1.0f / ker_norms.view({batch_size, 1, 1}));
        result.index_put_({mask}, 0.0f);

    } else {
        // Standard CC
        auto spec = img_fft * ker_fft.conj();
        result = torch::fft::irfft2(spec, {H, W});
    }

    // 6. Crop Output
    // The correlation result is cyclic. The valid part depends on how we define the anchor.
    // If kernel is at (0,0), then index (y,x) in output corresponds to correlation
    // when kernel top-left is at (y,x).
    // The valid output size for "valid" mode correlation (no padding) is (H-h+1, W-w+1).
    // The CUDA implementation crops:
    // crop_h = img_h - kernel_h + 1;
    // return full_result.slice(1, 0, crop_h).slice(2, 0, crop_w)

    int64_t crop_h = H - h + 1;
    int64_t crop_w = W - w + 1;

    result = result.slice(1, 0, crop_h).slice(2, 0, crop_w);

    // Reshape back
    return result.reshape(change_h_w_shapes(image, crop_h, crop_w));
}
