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
#include <type_traits>
#include "fft_cc.h"

// ==================================================================================
// MACROS & UTILS
// ==================================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      TORCH_CHECK(false, "CUDA failure"); \
    } \
  } while(0)

#define CUFFT_CHECK(call) do { \
    cufftResult res = (call); \
    if (res != CUFFT_SUCCESS) { \
      printf("CUFFT error %d at %s:%d\n", (int)res, __FILE__, __LINE__); \
      TORCH_CHECK(false, "CUFFT failure"); \
    } \
  } while(0)

// Key: H, W, Batch, Type, InPlace
using PlanKey = std::tuple<int, int, int, int, bool>; 

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
    
    cufftHandle get_plan(int height, int width, int batch_size, cufftType type, bool inplace) {
        std::lock_guard<std::mutex> lock(mutex_);
        int type_id = (int)type;
        PlanKey key = std::make_tuple(height, width, batch_size, type_id, inplace);
        
        if (cache_.find(key) != cache_.end()) return cache_[key];

        cufftHandle plan;
        int n[] = {height, width};
        int width_complex = width / 2 + 1;

        // Setup for Multi-Batch Plan
        if (inplace) {
             // In-Place Layout: Real [H, 2*W_c], Complex [H, W_c] share memory.
             int width_real_padded = 2 * width_complex;
             
             // Stride config for Reals (input of R2C, output of C2R)
             int real_embed[] = {height, width_real_padded}; 
             int real_dist = height * width_real_padded; 

             // Stride config for Complex
             int comp_embed[] = {height, width_complex};
             int comp_dist = height * width_complex;

             if (type == CUFFT_R2C || type == CUFFT_D2Z) {
                 CUFFT_CHECK(cufftPlanMany(&plan, 2, n, 
                    real_embed, 1, real_dist, // Input (Real)
                    comp_embed, 1, comp_dist, // Output (Complex)
                    type, batch_size));
             } else {
                 CUFFT_CHECK(cufftPlanMany(&plan, 2, n, 
                    comp_embed, 1, comp_dist, // Input (Complex)
                    real_embed, 1, real_dist, // Output (Real)
                    type, batch_size));
             }

        } else {
            // Out-of-place (Legacy/Standard Tightly Packed)
            if (type == CUFFT_R2C || type == CUFFT_D2Z) {
                 int inembed[] = {height, width};
                 int onembed[] = {height, width_complex};
                 int idist = height * width;
                 int odist = height * width_complex;
                 CUFFT_CHECK(cufftPlanMany(&plan, 2, n, inembed, 1, idist, onembed, 1, odist, type, batch_size));
            }
            else {
                 int inembed[] = {height, width_complex};
                 int onembed[] = {height, width};
                 int idist = height * width_complex;
                 int odist = height * width;
                 CUFFT_CHECK(cufftPlanMany(&plan, 2, n, inembed, 1, idist, onembed, 1, odist, type, batch_size));
            }
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

// Packs Image and Kernel into In-Place Buffer (Strided)
template <typename scalar_t>
__global__ void pack_fwd_inplace_kernel(
    const scalar_t* __restrict__ img_ptr,
    const scalar_t* __restrict__ ker_ptr,
    scalar_t* __restrict__ buffer_ptr,
    int H, int W, int h, int w,
    int stride_img, int stride_ker, 
    int row_stride_buffer, int batch_stride_buffer,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Pixel Index
    int b = blockIdx.z;
    if (b >= batch_size) return;
    if (idx >= H * W) return;

    int r = idx / W;
    int c = idx % W;

    int base_buffer = b * 2 * batch_stride_buffer;

    // Plane 0: Image
    // Write to Padded Buffer: r * row_stride + c
    int flat_buffer_idx = r * row_stride_buffer + c;
    
    buffer_ptr[base_buffer + flat_buffer_idx] = img_ptr[b * stride_img + idx];

    // Plane 1: Kernel
    scalar_t val = 0;
    if (r < h && c < w) {
        val = ker_ptr[b * stride_ker + r * w + c];
    }
    buffer_ptr[base_buffer + batch_stride_buffer + flat_buffer_idx] = val;
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

    ComplexT res = Traits::mul(I, Traits::conj(K));
    data_ptr[base + idx] = Traits::make_complex(Traits::real(res) * scale, Traits::imag(res) * scale);
}

// Finalize: Crop Output from Plane 0 (Padded Buffer)
template <typename scalar_t>
__global__ void finalize_fwd_inplace_kernel(
    const scalar_t* __restrict__ buffer_ptr,
    scalar_t* __restrict__ out_ptr,
    int H, int W, int out_h, int out_w,
    int row_stride_buffer, int batch_stride_buffer, int stride_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;
    if (idx >= out_h * out_w) return;

    int r = idx / out_w;
    int c = idx % out_w;
    
    // Read from Padded Buffer
    int buffer_idx = r * row_stride_buffer + c;
    out_ptr[b * stride_out + idx] = buffer_ptr[b * 2 * batch_stride_buffer + buffer_idx];
}

// ==================================================================================
// KERNELS: BACKWARD
// ==================================================================================

// Packs I, K, G into Padded Buffer
template <typename scalar_t>
__global__ void pack_bwd_inplace_kernel(
    const scalar_t* __restrict__ img_ptr,
    const scalar_t* __restrict__ ker_ptr,
    const scalar_t* __restrict__ grad_ptr,
    scalar_t* __restrict__ buffer_ptr,
    int H, int W, int h, int w, int out_h, int out_w,
    int stride_img, int stride_ker, int stride_grad,
    int row_stride_buffer, int batch_stride_buffer,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;
    if (idx >= H * W) return;

    int r = idx / W;
    int c = idx % W;


    // Buffer Layout:
    // Plane 0..B-1: Image
    // Plane B..2B-1: Kernel
    // Plane 2B..3B-1: Grad
    // Stride between planes is `batch_stride_buffer`.
    
    int flat_buffer_idx = r * row_stride_buffer + c;

    // 1. Image -> Plane b
    int p_img = b;
    buffer_ptr[p_img * batch_stride_buffer + flat_buffer_idx] = img_ptr[b * stride_img + idx];

    // 2. Kernel -> Plane B + b
    int p_ker = batch_size + b;
    scalar_t val_k = 0;
    if (r < h && c < w) {
        val_k = ker_ptr[b * stride_ker + r * w + c];
    }
    buffer_ptr[p_ker * batch_stride_buffer + flat_buffer_idx] = val_k;

    // 3. Grad -> Plane 2B + b
    int p_grad = 2 * batch_size + b;
    scalar_t val_g = 0;
    if (r < out_h && c < out_w) {
        val_g = grad_ptr[b * stride_grad + r * out_w + c];
    }
    buffer_ptr[p_grad * batch_stride_buffer + flat_buffer_idx] = val_g;
}

template <typename scalar_t>
__global__ void spectral_bwd_inplace_kernel(
    typename FFTTraits<scalar_t>::ComplexType* __restrict__ data_ptr,
    scalar_t scale,
    int plane_stride, // H * W_c
    double num_planes, // usually B
    int batch_size
) {
    using Traits = FFTTraits<scalar_t>;
    using ComplexT = typename Traits::ComplexType;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;
    if (idx >= plane_stride) return;

    // Map according to packing:
    // I: Plane b
    // K: Plane B + b
    // G: Plane 2B + b
    
    int B = batch_size;
    
    ComplexT* I_ptr = data_ptr + (b) * plane_stride;
    ComplexT* K_ptr = data_ptr + (B + b) * plane_stride;
    ComplexT* G_ptr = data_ptr + (2 * B + b) * plane_stride;
    
    ComplexT I = I_ptr[idx];
    ComplexT K = K_ptr[idx];
    ComplexT G = G_ptr[idx];

    // dI = G * K
    ComplexT dI = Traits::mul(G, K);
    
    // dK = I * conj(G)
    ComplexT dK = Traits::mul(I, Traits::conj(G));
    
    // Store back
    // Overwrite I with dI
    I_ptr[idx] = Traits::make_complex(Traits::real(dI)*scale, Traits::imag(dI)*scale);
    
    // Overwrite K with dK
    K_ptr[idx] = Traits::make_complex(Traits::real(dK)*scale, Traits::imag(dK)*scale);
}

template <typename scalar_t>
__global__ void finalize_bwd_inplace_kernel(
    const scalar_t* __restrict__ buffer_ptr,
    scalar_t* __restrict__ dI_ptr,
    scalar_t* __restrict__ dK_ptr,
    int H, int W, int h, int w,
    int row_stride_buffer, int batch_stride_buffer,
    int stride_dI, int stride_dK,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;
    if (idx >= H * W) return;

    int r = idx / W;
    int c = idx % W;
    
    int flat_idx = r * row_stride_buffer + c;
    int B = batch_size;

    // dI from Plane b
    dI_ptr[b * stride_dI + idx] = buffer_ptr[b * batch_stride_buffer + flat_idx];

    // dK from Plane B + b (Cropped)
    if (r < h && c < w) {
        dK_ptr[b * stride_dK + r * w + c] = buffer_ptr[(B + b) * batch_stride_buffer + flat_idx];
    }
}

// ==================================================================================
// HOST IMPL
// ==================================================================================

template <typename scalar_t>
torch::Tensor _fft_cc_forward_cuda_impl(torch::Tensor image, torch::Tensor kernel) {
    TORCH_CHECK(image.dim() >= 2, "Image/Kernel >= 2D");
    int H = image.size(-2);
    int W = image.size(-1);
    int h = kernel.size(-2);
    int w = kernel.size(-1);
    int B = image.numel() / (H*W);
    
    auto image_contig = image.reshape({B, H, W}).contiguous();
    auto kernel_contig = kernel.reshape({B, h, w}).contiguous();

    int out_h = H - h + 1;
    int out_w = W - w + 1;
    int W_c = W / 2 + 1;
    
    // In-Place Memory Layout
    // We allocate enough for COMPLEX [B*2, H, W_c]
    // Treated as Real [B*2, H, 2*W_c]
    auto complex_opts = image.options().dtype(
            std::is_same<scalar_t, double>::value ? torch::kComplexDouble : torch::kComplexFloat
    );
    auto buffer = torch::empty({B * 2, H, W_c}, complex_opts);
    
    scalar_t* buffer_real_ptr = (scalar_t*)buffer.data_ptr();
    
    // Strides in REALS
    int row_stride_real = 2 * W_c;
    int batch_stride_real = H * row_stride_real; 

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(image.device().index());

    // 1. Pack
    int threads = 256;
    int blocks = (H * W + threads - 1) / threads;
    
    pack_fwd_inplace_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
        image_contig.data_ptr<scalar_t>(),
        kernel_contig.data_ptr<scalar_t>(),
        buffer_real_ptr,
        H, W, h, w,
        H*W, h*w,
        row_stride_real, batch_stride_real,
        B
    );

    // 2. FFT (In-Place)
    using Traits = FFTTraits<scalar_t>;
    // Note: Plan uses B * 2
    cufftHandle plan_r2c = CuFFTPlanCache::instance().get_plan(H, W, B * 2, Traits::R2C_TYPE, true);
    CUFFT_CHECK(cufftSetStream(plan_r2c, stream));

    // Input and Output same pointer
    if constexpr (std::is_same<scalar_t, float>::value) {
        CUFFT_CHECK(cufftExecR2C(plan_r2c, buffer_real_ptr, (cufftComplex*)buffer.data_ptr()));
    } else {
        CUFFT_CHECK(cufftExecD2Z(plan_r2c, buffer_real_ptr, (cufftDoubleComplex*)buffer.data_ptr()));
    }

    // 3. Spectral
    int total_spec = H * W_c;
    blocks = (total_spec + threads - 1) / threads;
    scalar_t scale = 1.0f / (H * W);
    int plane_stride_complex = H * W_c;
    
    spectral_fwd_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
        (typename Traits::ComplexType*)buffer.data_ptr(),
        scale, plane_stride_complex, B
    );

    // 4. IFFT (In-Place)
    cufftHandle plan_c2r = CuFFTPlanCache::instance().get_plan(H, W, B * 2, Traits::C2R_TYPE, true);
    CUFFT_CHECK(cufftSetStream(plan_c2r, stream));

    if constexpr (std::is_same<scalar_t, float>::value) {
        CUFFT_CHECK(cufftExecC2R(plan_c2r, (cufftComplex*)buffer.data_ptr(), buffer_real_ptr));
    } else {
        CUFFT_CHECK(cufftExecZ2D(plan_c2r, (cufftDoubleComplex*)buffer.data_ptr(), buffer_real_ptr));
    }

    // 5. Finalize
    auto out = torch::empty({B, out_h, out_w}, image.options());
    blocks = (out_h * out_w + threads - 1) / threads;

    finalize_fwd_inplace_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
        buffer_real_ptr,
        out.data_ptr<scalar_t>(),
        H, W, out_h, out_w,
        row_stride_real, batch_stride_real, out_h * out_w,
        B
    );

    return out.contiguous().reshape(change_h_w_shapes(image, out_h, out_w));
}

torch::Tensor fft_cc_forward_cuda(torch::Tensor image, torch::Tensor kernel) {
    return AT_DISPATCH_SWITCH(image.scalar_type(), "fft_cc_forward_cuda",
        AT_DISPATCH_CASE(at::kFloat, [&] { return _fft_cc_forward_cuda_impl<float>(image, kernel); })
        AT_DISPATCH_CASE(at::kDouble, [&] { return _fft_cc_forward_cuda_impl<double>(image, kernel); })
    );
}

// Backward
template <typename scalar_t>
std::vector<torch::Tensor> _fft_cc_backward_cuda_impl(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel) {
    // ... Checks ...
    TORCH_CHECK(image.dim() >= 2, "Image >= 2D");
    int H = image.size(-2);
    int W = image.size(-1);
    int h = kernel.size(-2);
    int w = kernel.size(-1);
    int B = image.numel() / (H*W);
    
    int out_h = H - h + 1;
    int out_w = W - w + 1;
    int W_c = W / 2 + 1;

    auto image_contig = image.reshape({B, H, W}).contiguous();
    auto kernel_contig = kernel.reshape({B, h, w}).contiguous();
    auto grad_contig = grad_output.reshape({B, out_h, out_w}).contiguous();

    // In-Place Buffer: 3 * B planes
    auto complex_opts = image.options().dtype(
            std::is_same<scalar_t, double>::value ? torch::kComplexDouble : torch::kComplexFloat
    );
    auto buffer = torch::empty({B * 3, H, W_c}, complex_opts);
    scalar_t* buffer_real_ptr = (scalar_t*)buffer.data_ptr();
    
    int row_stride_real = 2 * W_c;
    int batch_stride_real = H * row_stride_real;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(image.device().index());

    // 1. Pack
    int threads = 256;
    int blocks = (H * W + threads - 1) / threads;

    pack_bwd_inplace_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
        image_contig.data_ptr<scalar_t>(),
        kernel_contig.data_ptr<scalar_t>(),
        grad_contig.data_ptr<scalar_t>(),
        buffer_real_ptr,
        H, W, h, w, out_h, out_w,
        H*W, h*w, out_h*out_w,
        row_stride_real, batch_stride_real, B
    );

    // 2. FFT
    using Traits = FFTTraits<scalar_t>;
    cufftHandle plan_r2c = CuFFTPlanCache::instance().get_plan(H, W, B * 3, Traits::R2C_TYPE, true);
    CUFFT_CHECK(cufftSetStream(plan_r2c, stream));

    if constexpr (std::is_same<scalar_t, float>::value) {
        CUFFT_CHECK(cufftExecR2C(plan_r2c, buffer_real_ptr, (cufftComplex*)buffer.data_ptr()));
    } else {
        CUFFT_CHECK(cufftExecD2Z(plan_r2c, buffer_real_ptr, (cufftDoubleComplex*)buffer.data_ptr()));
    }

    // 3. Spectral
    int total_spec = H * W_c;
    blocks = (total_spec + threads - 1) / threads;
    scalar_t scale = 1.0f / (H * W);
    int plane_stride_complex = H * W_c;

    spectral_bwd_inplace_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
        (typename Traits::ComplexType*)buffer.data_ptr(),
        scale, plane_stride_complex, 
        (double)0.0, // unused
        B
    );

    // 4. IFFT (Only first 2B planes: dI, dK)
    cufftHandle plan_c2r = CuFFTPlanCache::instance().get_plan(H, W, B * 2, Traits::C2R_TYPE, true);
    CUFFT_CHECK(cufftSetStream(plan_c2r, stream));

    if constexpr (std::is_same<scalar_t, float>::value) {
        CUFFT_CHECK(cufftExecC2R(plan_c2r, (cufftComplex*)buffer.data_ptr(), buffer_real_ptr));
    } else {
        CUFFT_CHECK(cufftExecZ2D(plan_c2r, (cufftDoubleComplex*)buffer.data_ptr(), buffer_real_ptr));
    }

    // 5. Finalize
    auto dI = torch::empty({B, H, W}, image.options());
    auto dK = torch::empty({B, h, w}, image.options());
    blocks = (H * W + threads - 1) / threads;

    finalize_bwd_inplace_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
        buffer_real_ptr,
        dI.data_ptr<scalar_t>(),
        dK.data_ptr<scalar_t>(),
        H, W, h, w,
        row_stride_real, batch_stride_real,
        H*W, h*w, B
    );

    return std::vector<torch::Tensor>{dI.contiguous().reshape(image.sizes()), dK.contiguous().reshape(kernel.sizes())};
}

std::vector<torch::Tensor> fft_cc_backward_cuda(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel) {
    return AT_DISPATCH_SWITCH(image.scalar_type(), "fft_cc_backward_cuda",
        AT_DISPATCH_CASE(at::kFloat, [&] { return _fft_cc_backward_cuda_impl<float>(grad_output, image, kernel); })
        AT_DISPATCH_CASE(at::kDouble, [&] { return _fft_cc_backward_cuda_impl<double>(grad_output, image, kernel); })
    );
}
