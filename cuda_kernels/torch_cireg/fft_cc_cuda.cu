#include <torch/extension.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuComplex.h>
#include <map>
#include <list>
#include <mutex>
#include <tuple>
#include <vector>
#include <iostream>
#include <cmath> 
#include "fft_cc.h"

// ==================================================================================
// MACROS & UTILS
// ==================================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error " << cudaGetErrorString(err) \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      TORCH_CHECK(false, "CUDA failure"); \
    } \
  } while(0)

#define CUFFT_CHECK(call) do { \
    cufftResult res = (call); \
    if (res != CUFFT_SUCCESS) { \
      std::cerr << "CUFFT error " << res << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      TORCH_CHECK(false, "CUFFT failure"); \
    } \
  } while(0)

using PlanKey = std::tuple<int, int, int, int>; // H, W, Batch, Type

// ==================================================================================
// TRAITS
// ==================================================================================
template <typename T> struct FFTTraits;

template <> struct FFTTraits<float> {
    using ComplexType = cufftComplex;
    static const cufftType R2C_TYPE = CUFFT_R2C;
    static const cufftType C2R_TYPE = CUFFT_C2R;
    __device__ static ComplexType make_complex(float r, float i) { return make_cuComplex(r, i); }
    __device__ static float real(ComplexType c) { return cuCrealf(c); }
    __device__ static float imag(ComplexType c) { return cuCimagf(c); }
    __device__ static ComplexType conj(ComplexType c) { return make_cuComplex(cuCrealf(c), -cuCimagf(c)); }
    __device__ static ComplexType mul(ComplexType a, ComplexType b) { return cuCmulf(a, b); }
};

template <> struct FFTTraits<double> {
    using ComplexType = cufftDoubleComplex;
    static const cufftType R2C_TYPE = CUFFT_D2Z;
    static const cufftType C2R_TYPE = CUFFT_Z2D;
    __device__ static ComplexType make_complex(double r, double i) { return make_cuDoubleComplex(r, i); }
    __device__ static double real(ComplexType c) { return cuCreal(c); }
    __device__ static double imag(ComplexType c) { return cuCimag(c); }
    __device__ static ComplexType conj(ComplexType c) { return make_cuDoubleComplex(cuCreal(c), -cuCimag(c)); }
    __device__ static ComplexType mul(ComplexType a, ComplexType b) { return cuCmul(a, b); }
};

// ==================================================================================
// PLAN CACHE
// ==================================================================================
class CuFFTPlanCache {
public:
    static CuFFTPlanCache& instance() { static CuFFTPlanCache i; return i; }
    
    cufftHandle get_plan(int height, int width, int batch_size, cufftType type) {
        std::lock_guard<std::mutex> lock(mutex_);
        int type_id = (int)type;
        PlanKey key = std::make_tuple(height, width, batch_size, type_id);
        
        if (cache_.find(key) != cache_.end()) return cache_[key];

        cufftHandle plan;
        int n[] = {height, width};
        int width_complex = width / 2 + 1;

        // Setup for Multi-Batch Plan
        // R2C / D2Z
        if (type == CUFFT_R2C || type == CUFFT_D2Z) {
             int inembed[] = {height, width};
             int onembed[] = {height, width_complex};
             int idist = height * width;
             int odist = height * width_complex;
             CUFFT_CHECK(cufftPlanMany(&plan, 2, n, inembed, 1, idist, onembed, 1, odist, type, batch_size));
        }
        // C2R / Z2D
        else {
             int inembed[] = {height, width_complex};
             int onembed[] = {height, width};
             int idist = height * width_complex;
             int odist = height * width;
             CUFFT_CHECK(cufftPlanMany(&plan, 2, n, inembed, 1, idist, onembed, 1, odist, type, batch_size));
        }

        cache_[key] = plan;
        return plan;
    }
private:
    std::map<PlanKey, cufftHandle> cache_;
    std::mutex mutex_;
};

// ==================================================================================
// KERNELS: FORWARD
// ==================================================================================

// Packs Image (Plane 0) and Kernel (Plane 1) into Arena
template <typename scalar_t>
__global__ void pack_fwd_kernel(
    const scalar_t* __restrict__ img_ptr,
    const scalar_t* __restrict__ ker_ptr,
    scalar_t* __restrict__ arena_ptr,
    int H, int W, int h, int w,
    int stride_img, int stride_ker, int stride_arena,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;

    // We process by pixels.
    // Image pixels: H*W
    // Kernel pixels: h*w
    // We can map idx to either image or kernel based on range,
    // or just run two separate launches. Merged launch is complex for different sizes.
    // Let's assume idx covers H*W (max size).
    
    // Pack Image
    if (idx < H*W) {
        // Plane 0
        arena_ptr[b * 2 * stride_arena + idx] = img_ptr[b * stride_img + idx];
    }
    
    // Pack Kernel
    if (idx < h*w) {
        int r = idx / w;
        int c = idx % w;
        int arena_idx = r * W + c; // Pad to H, W
        // Plane 1
        arena_ptr[b * 2 * stride_arena + stride_arena + arena_idx] = ker_ptr[b * stride_ker + idx];
    }
}

// Complex Math: O = I * conj(K)
template <typename scalar_t>
__global__ void spectral_fwd_kernel(
    typename FFTTraits<scalar_t>::ComplexType* __restrict__ data_ptr,
    scalar_t scale,
    int plane_stride, // H * W_c
    int batch_size
) {
    using Traits = FFTTraits<scalar_t>;
    using ComplexT = typename Traits::ComplexType;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;
    if (idx >= plane_stride) return;

    int base = b * 2 * plane_stride;
    ComplexT I = data_ptr[base + idx];
    ComplexT K = data_ptr[base + plane_stride + idx];

    // O = I * conj(K) * scale
    ComplexT res = Traits::mul(I, Traits::conj(K));

    // Store in Plane 0
    data_ptr[base + idx] = Traits::make_complex(Traits::real(res) * scale, Traits::imag(res) * scale);
}

// Finalize: Crop Output from Plane 0
template <typename scalar_t>
__global__ void finalize_fwd_kernel(
    const scalar_t* __restrict__ arena_ptr,
    scalar_t* __restrict__ out_ptr,
    int H, int W, int out_h, int out_w,
    int stride_arena, int stride_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;
    if (idx >= stride_out) return; // out_h * out_w

    int r = idx / out_w;
    int c = idx % out_w;
    int arena_idx = r * W + c;

    out_ptr[b * stride_out + idx] = arena_ptr[b * 2 * stride_arena + arena_idx];
}

// ==================================================================================
// KERNELS: BACKWARD
// ==================================================================================

// Packs I (P0), K (P1), G (P2)
template <typename scalar_t>
__global__ void pack_bwd_kernel(
    const scalar_t* __restrict__ img_ptr,
    const scalar_t* __restrict__ ker_ptr,
    const scalar_t* __restrict__ grad_ptr,
    scalar_t* __restrict__ arena_ptr,
    int H, int W, int h, int w, int out_h, int out_w,
    int stride_img, int stride_ker, int stride_grad, int stride_arena,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;

    int base_arena = b * 3 * stride_arena;

    // Pack Image (Plane 0)
    if (idx < H*W) {
        arena_ptr[base_arena + idx] = img_ptr[b * stride_img + idx];
    }

    // Pack Kernel (Plane 1)
    if (idx < h*w) {
        int r = idx / w;
        int c = idx % w;
        int arena_idx = r * W + c;
        arena_ptr[base_arena + stride_arena + arena_idx] = ker_ptr[b * stride_ker + idx];
    }

    // Pack Grad (Plane 2)
    if (idx < out_h * out_w) {
        int r = idx / out_w;
        int c = idx % out_w;
        int arena_idx = r * W + c;
        arena_ptr[base_arena + 2 * stride_arena + arena_idx] = grad_ptr[b * stride_grad + idx];
    }
}

// Spectral BWD:
// dI (P0) = G * K
// dK (P1) = I * conj(G)
template <typename scalar_t>
__global__ void spectral_bwd_kernel(
    typename FFTTraits<scalar_t>::ComplexType* __restrict__ data_ptr,
    scalar_t scale,
    int plane_stride,
    int batch_size
) {
    using Traits = FFTTraits<scalar_t>;
    using ComplexT = typename Traits::ComplexType;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;
    if (idx >= plane_stride) return;

    int base = b * 3 * plane_stride;
    
    ComplexT I = data_ptr[base + 0 * plane_stride + idx];
    ComplexT K = data_ptr[base + 1 * plane_stride + idx];
    ComplexT G = data_ptr[base + 2 * plane_stride + idx];

    // dI = G * K
    ComplexT dI = Traits::mul(G, K);

    // dK = I * conj(G)
    ComplexT dK = Traits::mul(I, Traits::conj(G));
    
    // Store Back
    data_ptr[base + 0 * plane_stride + idx] = Traits::make_complex(Traits::real(dI)*scale, Traits::imag(dI)*scale);
    data_ptr[base + 1 * plane_stride + idx] = Traits::make_complex(Traits::real(dK)*scale, Traits::imag(dK)*scale);
}

// Finalize BWD: Crop dI (P0) and dK (P1)
template <typename scalar_t>
__global__ void finalize_bwd_kernel(
    const scalar_t* __restrict__ arena_ptr,
    scalar_t* __restrict__ dI_ptr,
    scalar_t* __restrict__ dK_ptr,
    int H, int W, int h, int w,
    int stride_arena, int stride_dI, int stride_dK,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;

    // Handle dI (Full Size H*W)
    if (idx < H*W) {
        // Plane 0
        dI_ptr[b * stride_dI + idx] = arena_ptr[b * 3 * stride_arena + idx];
    }

    // Handle dK (Crop to h*w)
    if (idx < h*w) {
        int r = idx / w;
        int c = idx % w;
        int arena_idx = r * W + c;
        // Plane 1
        dK_ptr[b * stride_dK + idx] = arena_ptr[b * 3 * stride_arena + stride_arena + arena_idx];
    }
}

// ==================================================================================
// CUDA HOST IMPLEMENTATIONS
// ==================================================================================

torch::Tensor fft_cc_forward_cuda(torch::Tensor image, torch::Tensor kernel) {
    return AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "fft_cc_forward_cuda", [&] {
        int H = image.size(-2);
        int W = image.size(-1);
        int h = kernel.size(-2);
        int w = kernel.size(-1);
        int B = image.numel() / (H*W);

        int out_h = H - h + 1;
        int out_w = W - w + 1;

        // Allocation
        // Arena: [B, 2, H, W]
        auto arena = torch::zeros({B * 2, H, W}, image.options());
        int W_c = W / 2 + 1;

        auto complex_opts = image.options().dtype(
             std::is_same<scalar_t, double>::value ? torch::kComplexDouble : torch::kComplexFloat
        );
        auto arena_c = torch::empty({B * 2, H, W_c}, complex_opts);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream(image.device().index());

        // 1. Pack
        int threads = 256;
        int total_pixels = H * W; // Max pixels to cover
        int blocks = (total_pixels + threads - 1) / threads;
        
        pack_fwd_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
            image.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            arena.data_ptr<scalar_t>(),
            H, W, h, w,
            H*W, h*w, H*W, B
        );

        // 2. FFT
        using Traits = FFTTraits<scalar_t>;
        cufftHandle plan_r2c = CuFFTPlanCache::instance().get_plan(H, W, B * 2, Traits::R2C_TYPE);
        CUFFT_CHECK(cufftSetStream(plan_r2c, stream));

        if constexpr (std::is_same<scalar_t, float>::value) {
            CUFFT_CHECK(cufftExecR2C(plan_r2c, arena.data_ptr<scalar_t>(), (cufftComplex*)arena_c.data_ptr()));
        } else {
            CUFFT_CHECK(cufftExecD2Z(plan_r2c, arena.data_ptr<scalar_t>(), (cufftDoubleComplex*)arena_c.data_ptr()));
        }

        // 3. Spectral
        int total_spec = H * W_c;
        blocks = (total_spec + threads - 1) / threads;
        scalar_t scale = 1.0f / (H * W);
        
        spectral_fwd_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
            (typename Traits::ComplexType*)arena_c.data_ptr(),
            scale, total_spec, B
        );

        // 4. IFFT (Only Plane 0 needed, but batch is 2? No, we only need Plane 0)
        // Wait, cufftExecC2R expects contiguous batch.
        // If we want only Plane 0, we can run IFFT on batch B, stride 2 * plane_stride?
        // Or just run IFFT on B*2 and ignore Plane 1 output. Running on B*2 is simpler given the plan.

        cufftHandle plan_c2r = CuFFTPlanCache::instance().get_plan(H, W, B * 2, Traits::C2R_TYPE);
        CUFFT_CHECK(cufftSetStream(plan_c2r, stream));

        if constexpr (std::is_same<scalar_t, float>::value) {
            CUFFT_CHECK(cufftExecC2R(plan_c2r, (cufftComplex*)arena_c.data_ptr(), arena.data_ptr<scalar_t>()));
        } else {
            CUFFT_CHECK(cufftExecZ2D(plan_c2r, (cufftDoubleComplex*)arena_c.data_ptr(), arena.data_ptr<scalar_t>()));
        }

        // 5. Finalize
        auto out = torch::empty({B, out_h, out_w}, image.options());
        int total_out = out_h * out_w;
        blocks = (total_out + threads - 1) / threads;

        finalize_fwd_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
            arena.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            H, W, out_h, out_w,
            H*W, out_h*out_w, B
        );

        return out.reshape(change_h_w_shapes(image, out_h, out_w));
    });
}

std::vector<torch::Tensor> fft_cc_backward_cuda(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel) {
    return AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "fft_cc_backward_cuda", [&] {
        int H = image.size(-2);
        int W = image.size(-1);
        int h = kernel.size(-2);
        int w = kernel.size(-1);
        int B = image.numel() / (H*W);
        int out_h = H - h + 1;
        int out_w = W - w + 1;

        // Arena: [B, 3, H, W]
        auto arena = torch::zeros({B * 3, H, W}, image.options());
        int W_c = W / 2 + 1;

        auto complex_opts = image.options().dtype(
             std::is_same<scalar_t, double>::value ? torch::kComplexDouble : torch::kComplexFloat
        );
        auto arena_c = torch::empty({B * 3, H, W_c}, complex_opts);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream(image.device().index());

        // 1. Pack
        int threads = 256;
        int total_pixels = H * W;
        int blocks = (total_pixels + threads - 1) / threads;

        pack_bwd_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
            image.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            arena.data_ptr<scalar_t>(),
            H, W, h, w, out_h, out_w,
            H*W, h*w, out_h*out_w, H*W, B
        );

        // 2. FFT
        using Traits = FFTTraits<scalar_t>;
        cufftHandle plan_r2c = CuFFTPlanCache::instance().get_plan(H, W, B * 3, Traits::R2C_TYPE);
        CUFFT_CHECK(cufftSetStream(plan_r2c, stream));

        if constexpr (std::is_same<scalar_t, float>::value) {
            CUFFT_CHECK(cufftExecR2C(plan_r2c, arena.data_ptr<scalar_t>(), (cufftComplex*)arena_c.data_ptr()));
        } else {
            CUFFT_CHECK(cufftExecD2Z(plan_r2c, arena.data_ptr<scalar_t>(), (cufftDoubleComplex*)arena_c.data_ptr()));
        }

        // 3. Spectral
        int total_spec = H * W_c;
        blocks = (total_spec + threads - 1) / threads;
        scalar_t scale = 1.0f / (H * W);

        spectral_bwd_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
            (typename Traits::ComplexType*)arena_c.data_ptr(),
            scale, total_spec, B
        );

        // 4. IFFT (Need P0 and P1. We run on all 3 for simplicity because plan is cached, or make new plan for 2?)
        // To save time, we should probably run IFFT on B*2 if we can slice.
        // But memory is contiguous B*3.
        // We can create a plan for B*3 and just ignore P2.
        // Or we can create a plan with stride/dist to skip P2?
        // Simpler: Just run IFFT on all 3.

        cufftHandle plan_c2r = CuFFTPlanCache::instance().get_plan(H, W, B * 3, Traits::C2R_TYPE);
        CUFFT_CHECK(cufftSetStream(plan_c2r, stream));

        if constexpr (std::is_same<scalar_t, float>::value) {
            CUFFT_CHECK(cufftExecC2R(plan_c2r, (cufftComplex*)arena_c.data_ptr(), arena.data_ptr<scalar_t>()));
        } else {
            CUFFT_CHECK(cufftExecZ2D(plan_c2r, (cufftDoubleComplex*)arena_c.data_ptr(), arena.data_ptr<scalar_t>()));
        }

        // 5. Finalize
        auto dI = torch::empty_like(image);
        auto dK = torch::empty_like(kernel);

        blocks = (total_pixels + threads - 1) / threads;

        finalize_bwd_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
            arena.data_ptr<scalar_t>(),
            dI.data_ptr<scalar_t>(),
            dK.data_ptr<scalar_t>(),
            H, W, h, w,
            H*W, H*W, h*w, B
        );

        return {dI, dK};
    });
}
