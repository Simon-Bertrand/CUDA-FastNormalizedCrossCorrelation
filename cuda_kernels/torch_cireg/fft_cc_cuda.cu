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

// Packs Image (Plane 0) and Kernel (Plane 1) into Arena, Handling Padding
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

    // We process by pixels of H*W (max)
    if (idx >= H*W) return;
    
    int r = idx / W;
    int c = idx % W;
    
    int base_arena = b * 2 * stride_arena;

    // Plane 0: Image (Full Copy, HxW -> HxW)
    // Assuming image matches HxW
    arena_ptr[base_arena + idx] = img_ptr[b * stride_img + idx];

    // Plane 1: Kernel (Copy hxw, Zero Pad rest)
    scalar_t val = 0;
    if (r < h && c < w) {
        // Map to flat kernel index
        int k_idx = r * w + c;
        val = ker_ptr[b * stride_ker + k_idx];
    }
    arena_ptr[base_arena + stride_arena + idx] = val;
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
// KERNELS: BACKWARD (SPLIT MEMORY)
// ==================================================================================

// Packs I (P0), K (P1) into IK buffer. G (P0) into G buffer.
// Zero pads as needed.
template <typename scalar_t>
__global__ void pack_bwd_split_kernel(
    const scalar_t* __restrict__ img_ptr,
    const scalar_t* __restrict__ ker_ptr,
    const scalar_t* __restrict__ grad_ptr,
    scalar_t* __restrict__ ik_ptr,
    scalar_t* __restrict__ g_ptr,
    int H, int W, int h, int w, int out_h, int out_w,
    int stride_img, int stride_ker, int stride_grad,
    int stride_ik_plane, int stride_g_plane,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;

    if (idx >= H*W) return;

    int r = idx / W;
    int c = idx % W;

    int base_ik = b * 2 * stride_ik_plane;
    int base_g  = b * 1 * stride_g_plane;

    // 1. Pack Image -> IK Plane 0
    ik_ptr[base_ik + idx] = img_ptr[b * stride_img + idx];

    // 2. Pack Kernel -> IK Plane 1 (Padded)
    scalar_t val_k = 0;
    if (r < h && c < w) {
        val_k = ker_ptr[b * stride_ker + r * w + c];
    }
    ik_ptr[base_ik + stride_ik_plane + idx] = val_k;

    // 3. Pack Grad -> G Plane 0 (Padded)
    scalar_t val_g = 0;
    if (r < out_h && c < out_w) {
        val_g = grad_ptr[b * stride_grad + r * out_w + c];
    }
    g_ptr[base_g + idx] = val_g;
}

// Spectral BWD:
// Reads IK and G. Writes dI, dK to IK.
// dI (P0) = G * K
// dK (P1) = I * conj(G)
template <typename scalar_t>
__global__ void spectral_bwd_split_kernel(
    typename FFTTraits<scalar_t>::ComplexType* __restrict__ ik_ptr,
    const typename FFTTraits<scalar_t>::ComplexType* __restrict__ g_ptr,
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

    int base_ik = b * 2 * plane_stride;
    int base_g  = b * 1 * plane_stride;
    
    ComplexT I = ik_ptr[base_ik + 0 * plane_stride + idx];
    ComplexT K = ik_ptr[base_ik + 1 * plane_stride + idx];
    ComplexT G = g_ptr[base_g + idx];

    // dI = G * K
    ComplexT dI = Traits::mul(G, K);

    // dK = I * conj(G)
    ComplexT dK = Traits::mul(I, Traits::conj(G));
    
    // Store Back
    ik_ptr[base_ik + 0 * plane_stride + idx] = Traits::make_complex(Traits::real(dI)*scale, Traits::imag(dI)*scale);
    ik_ptr[base_ik + 1 * plane_stride + idx] = Traits::make_complex(Traits::real(dK)*scale, Traits::imag(dK)*scale);
}

// Finalize BWD: Crop dI (P0) and dK (P1) from IK buffer
template <typename scalar_t>
__global__ void finalize_bwd_split_kernel(
    const scalar_t* __restrict__ ik_ptr,
    scalar_t* __restrict__ dI_ptr,
    scalar_t* __restrict__ dK_ptr,
    int H, int W, int h, int w,
    int stride_ik_plane, int stride_dI, int stride_dK,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;
    if (idx >= H*W) return;

    int r = idx / W;
    int c = idx % W;

    int base_ik = b * 2 * stride_ik_plane;

    // Handle dI (Full Size H*W) - Plane 0
    dI_ptr[b * stride_dI + idx] = ik_ptr[base_ik + idx];

    // Handle dK (Crop to h*w) - Plane 1
    if (r < h && c < w) {
        dK_ptr[b * stride_dK + r * w + c] = ik_ptr[base_ik + stride_ik_plane + idx];
    }
}

// ==================================================================================
// CUDA HOST IMPLEMENTATIONS
// ==================================================================================

torch::Tensor fft_cc_forward_cuda(torch::Tensor image, torch::Tensor kernel) {
    return AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "fft_cc_forward_cuda", [&] {
        TORCH_CHECK(image.dim() >= 2, "Image must have at least 2 dimensions");
        TORCH_CHECK(kernel.dim() >= 2, "Kernel must have at least 2 dimensions");

        int H = image.size(-2);
        int W = image.size(-1);
        int h = kernel.size(-2);
        int w = kernel.size(-1);
        int B = image.numel() / (H*W);
        TORCH_CHECK(kernel.numel() / (h*w) == B, "Batch size mismatch");

        // Ensure contiguous inputs for raw kernels
        auto image_contig = image.reshape({B, H, W}).contiguous();
        auto kernel_contig = kernel.reshape({B, h, w}).contiguous();

        int out_h = H - h + 1;
        int out_w = W - w + 1;

        // Allocation: empty to avoid zero init, kernel handles padding
        auto arena = torch::empty({B * 2, H, W}, image.options());
        int W_c = W / 2 + 1;

        auto complex_opts = image.options().dtype(
             std::is_same<scalar_t, double>::value ? torch::kComplexDouble : torch::kComplexFloat
        );
        auto arena_c = torch::empty({B * 2, H, W_c}, complex_opts);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream(image.device().index());

        // 1. Pack
        int threads = 256;
        int total_pixels = H * W;
        int blocks = (total_pixels + threads - 1) / threads;
        
        pack_fwd_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
            image_contig.data_ptr<scalar_t>(),
            kernel_contig.data_ptr<scalar_t>(),
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

        // 4. IFFT (Result in arena Plane 0)
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

        // Return contiguous output with correct shape
        return out.contiguous().reshape(change_h_w_shapes(image, out_h, out_w));
    });
}

std::vector<torch::Tensor> fft_cc_backward_cuda(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel) {
    return AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "fft_cc_backward_cuda", [&] {
        TORCH_CHECK(image.dim() >= 2, "Image must have at least 2 dimensions");
        TORCH_CHECK(kernel.dim() >= 2, "Kernel must have at least 2 dimensions");

        int H = image.size(-2);
        int W = image.size(-1);
        int h = kernel.size(-2);
        int w = kernel.size(-1);
        int B = image.numel() / (H*W);
        TORCH_CHECK(kernel.numel() / (h*w) == B, "Batch size mismatch");

        int out_h = H - h + 1;
        int out_w = W - w + 1;
        int W_c = W / 2 + 1;

        // Ensure contiguous inputs
        auto image_contig = image.reshape({B, H, W}).contiguous();
        auto kernel_contig = kernel.reshape({B, h, w}).contiguous();
        auto grad_contig = grad_output.reshape({B, out_h, out_w}).contiguous();

        // Allocations: Separate IK and G
        // Arena IK: [B, 2, H, W]
        auto arena_ik = torch::empty({B * 2, H, W}, image.options());
        // Arena G:  [B, 1, H, W]
        auto arena_g  = torch::empty({B * 1, H, W}, image.options());

        auto complex_opts = image.options().dtype(
             std::is_same<scalar_t, double>::value ? torch::kComplexDouble : torch::kComplexFloat
        );
        auto spec_ik = torch::empty({B * 2, H, W_c}, complex_opts);
        auto spec_g  = torch::empty({B * 1, H, W_c}, complex_opts);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream(image.device().index());

        // 1. Pack
        int threads = 256;
        int total_pixels = H * W;
        int blocks = (total_pixels + threads - 1) / threads;

        pack_bwd_split_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
            image_contig.data_ptr<scalar_t>(),
            kernel_contig.data_ptr<scalar_t>(),
            grad_contig.data_ptr<scalar_t>(),
            arena_ik.data_ptr<scalar_t>(),
            arena_g.data_ptr<scalar_t>(),
            H, W, h, w, out_h, out_w,
            H*W, h*w, out_h*out_w,
            H*W, H*W, // Stride planes
            B
        );

        // 2. FFTs
        using Traits = FFTTraits<scalar_t>;

        // IK FFT
        cufftHandle plan_r2c_ik = CuFFTPlanCache::instance().get_plan(H, W, B * 2, Traits::R2C_TYPE);
        CUFFT_CHECK(cufftSetStream(plan_r2c_ik, stream));
        if constexpr (std::is_same<scalar_t, float>::value) {
            CUFFT_CHECK(cufftExecR2C(plan_r2c_ik, arena_ik.data_ptr<scalar_t>(), (cufftComplex*)spec_ik.data_ptr()));
        } else {
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_ik, arena_ik.data_ptr<scalar_t>(), (cufftDoubleComplex*)spec_ik.data_ptr()));
        }

        // G FFT
        cufftHandle plan_r2c_g = CuFFTPlanCache::instance().get_plan(H, W, B * 1, Traits::R2C_TYPE);
        CUFFT_CHECK(cufftSetStream(plan_r2c_g, stream));
        if constexpr (std::is_same<scalar_t, float>::value) {
            CUFFT_CHECK(cufftExecR2C(plan_r2c_g, arena_g.data_ptr<scalar_t>(), (cufftComplex*)spec_g.data_ptr()));
        } else {
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_g, arena_g.data_ptr<scalar_t>(), (cufftDoubleComplex*)spec_g.data_ptr()));
        }

        // 3. Spectral
        int total_spec = H * W_c;
        blocks = (total_spec + threads - 1) / threads;
        scalar_t scale = 1.0f / (H * W);

        spectral_bwd_split_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
            (typename Traits::ComplexType*)spec_ik.data_ptr(),
            (typename Traits::ComplexType*)spec_g.data_ptr(),
            scale, total_spec, B
        );

        // 4. IFFT (Only on IK)
        cufftHandle plan_c2r_ik = CuFFTPlanCache::instance().get_plan(H, W, B * 2, Traits::C2R_TYPE);
        CUFFT_CHECK(cufftSetStream(plan_c2r_ik, stream));

        if constexpr (std::is_same<scalar_t, float>::value) {
            CUFFT_CHECK(cufftExecC2R(plan_c2r_ik, (cufftComplex*)spec_ik.data_ptr(), arena_ik.data_ptr<scalar_t>()));
        } else {
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_ik, (cufftDoubleComplex*)spec_ik.data_ptr(), arena_ik.data_ptr<scalar_t>()));
        }

        // 5. Finalize
        auto dI = torch::empty({B, H, W}, image.options());
        auto dK = torch::empty({B, h, w}, image.options());

        blocks = (total_pixels + threads - 1) / threads;

        finalize_bwd_split_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
            arena_ik.data_ptr<scalar_t>(),
            dI.data_ptr<scalar_t>(),
            dK.data_ptr<scalar_t>(),
            H, W, h, w,
            H*W, H*W, h*w, B
        );

        return {dI.contiguous().reshape(image.sizes()), dK.contiguous().reshape(kernel.sizes())};
    });
}
