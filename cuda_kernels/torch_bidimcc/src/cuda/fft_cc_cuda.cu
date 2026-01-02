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
        if (stride_ker == 0) {
            // Broadcast kernel (b=0)
            val = ker_ptr[r * w + c];
        } else {
            val = ker_ptr[b * stride_ker + r * w + c];
        }
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
// Dynamic packing based on flags (encoded in kernel logic via template or args?)
// To keep it simple, we pass flags as bool args.
// But we need to know which planes map to what.
// We pass destination plane indices. -1 if not used.
template <typename scalar_t>
__global__ void pack_bwd_inplace_kernel(
    const scalar_t* __restrict__ img_ptr,
    const scalar_t* __restrict__ ker_ptr,
    const scalar_t* __restrict__ grad_ptr,
    scalar_t* __restrict__ buffer_ptr,
    int H, int W, int h, int w, int out_h, int out_w,
    int stride_img, int stride_ker, int stride_grad,
    int row_stride_buffer, int batch_stride_buffer,
    int p_img_idx, int p_ker_idx, int p_grad_idx,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;
    if (idx >= H * W) return;

    int r = idx / W;
    int c = idx % W;

    int flat_buffer_idx = r * row_stride_buffer + c;

    // 1. Image
    if (p_img_idx >= 0) {
        int plane = p_img_idx * batch_size + b;
        buffer_ptr[plane * batch_stride_buffer + flat_buffer_idx] = img_ptr[b * stride_img + idx];
    }

    // 2. Kernel
    if (p_ker_idx >= 0) {
        int plane = p_ker_idx * batch_size + b;
        scalar_t val_k = 0;
        if (r < h && c < w) {
            if (stride_ker == 0) val_k = ker_ptr[r * w + c];
            else val_k = ker_ptr[b * stride_ker + r * w + c];
        }
        buffer_ptr[plane * batch_stride_buffer + flat_buffer_idx] = val_k;
    }

    // 3. Grad
    if (p_grad_idx >= 0) {
        int plane = p_grad_idx * batch_size + b;
        scalar_t val_g = 0;
        if (r < out_h && c < out_w) {
            val_g = grad_ptr[b * stride_grad + r * out_w + c];
        }
        buffer_ptr[plane * batch_stride_buffer + flat_buffer_idx] = val_g;
    }
}

template <typename scalar_t>
__global__ void spectral_bwd_inplace_kernel(
    typename FFTTraits<scalar_t>::ComplexType* __restrict__ data_ptr,
    scalar_t scale,
    int plane_stride, // H * W_c
    int p_img_idx, int p_ker_idx, int p_grad_idx,
    int p_dI_idx, int p_dK_idx,
    int batch_size
) {
    using Traits = FFTTraits<scalar_t>;
    using ComplexT = typename Traits::ComplexType;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (b >= batch_size) return;
    if (idx >= plane_stride) return;

    // Load inputs
    // If index is -1, means we don't have it (and logic shouldn't use it)
    int B = batch_size;
    
    ComplexT I, K, G;
    
    if (p_img_idx >= 0) I = data_ptr[(p_img_idx * B + b) * plane_stride + idx];
    if (p_ker_idx >= 0) K = data_ptr[(p_ker_idx * B + b) * plane_stride + idx];
    if (p_grad_idx >= 0) G = data_ptr[(p_grad_idx * B + b) * plane_stride + idx];

    if (p_dI_idx >= 0) {
        // dI = G * K
        ComplexT dI = Traits::mul(G, K);
        data_ptr[(p_dI_idx * B + b) * plane_stride + idx] = Traits::make_complex(Traits::real(dI)*scale, Traits::imag(dI)*scale);
    }
    
    if (p_dK_idx >= 0) {
        // dK = I * conj(G)
        ComplexT dK = Traits::mul(I, Traits::conj(G));
        data_ptr[(p_dK_idx * B + b) * plane_stride + idx] = Traits::make_complex(Traits::real(dK)*scale, Traits::imag(dK)*scale);
    }
}

template <typename scalar_t>
__global__ void finalize_bwd_inplace_kernel(
    const scalar_t* __restrict__ buffer_ptr,
    scalar_t* __restrict__ dI_ptr,
    scalar_t* __restrict__ dK_ptr,
    int H, int W, int h, int w,
    int row_stride_buffer, int batch_stride_buffer,
    int stride_dI, int stride_dK,
    int p_dI_idx, int p_dK_idx,
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

    if (p_dI_idx >= 0 && dI_ptr != nullptr) {
        dI_ptr[b * stride_dI + idx] = buffer_ptr[(p_dI_idx * B + b) * batch_stride_buffer + flat_idx];
    }

    if (p_dK_idx >= 0 && dK_ptr != nullptr) {
        if (r < h && c < w) {
            dK_ptr[b * stride_dK + r * w + c] = buffer_ptr[(p_dK_idx * B + b) * batch_stride_buffer + flat_idx];
        }
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
    int KB = kernel.numel() / (h*w);

    bool broadcast_kernel = (KB == 1 && B > 1);
    
    auto image_contig = image.reshape({B, H, W}).contiguous();
    auto kernel_contig = kernel.reshape({KB, h, w}).contiguous();

    int out_h = H - h + 1;
    int out_w = W - w + 1;
    int W_c = W / 2 + 1;
    
    // In-Place Memory Layout
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
        H*W, broadcast_kernel ? 0 : h*w,
        row_stride_real, batch_stride_real,
        B
    );

    // 2. FFT (In-Place)
    using Traits = FFTTraits<scalar_t>;
    // Note: Plan uses B * 2
    cufftHandle plan_r2c = CuFFTPlanCache::instance().get_plan(H, W, B * 2, Traits::R2C_TYPE, true);
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
std::vector<torch::Tensor> _fft_cc_backward_cuda_impl(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel, std::vector<bool> output_mask) {
    bool compute_dI = output_mask[0];
    bool compute_dK = output_mask[1];

    if (!compute_dI && !compute_dK) return {torch::Tensor(), torch::Tensor()};

    int h = kernel.size(-2);
    int w = kernel.size(-1);
    int H, W;
    
    if (image.defined()) {
        H = image.size(-2);
        W = image.size(-1);
    } else {
        int out_h = grad_output.size(-2);
        int out_w = grad_output.size(-1);
        H = out_h + h - 1;
        W = out_w + w - 1;
    }

    int out_h = H - h + 1;
    int out_w = W - w + 1;

    int B = grad_output.numel() / (out_h * out_w);
    int KB = kernel.numel() / (h*w);
    bool broadcast_kernel = (KB == 1 && B > 1);
    int W_c = W / 2 + 1;

    auto grad_contig = grad_output.reshape({B, out_h, out_w}).contiguous();

    // Determine mapping and planes
    int num_planes = 0;
    int p_img_idx = -1;
    int p_ker_idx = -1;
    int p_grad_idx = -1;

    int p_dI_idx = -1;
    int p_dK_idx = -1;

    if (compute_dI && compute_dK) {
        // I(0), K(1), G(2)
        num_planes = 3;
        p_img_idx = 0;
        p_ker_idx = 1;
        p_grad_idx = 2;

        p_dI_idx = 0; // Overwrite I
        p_dK_idx = 1; // Overwrite K
    } else if (compute_dI) {
        // G(0), K(1) -> dI(0)
        num_planes = 2;
        p_grad_idx = 0;
        p_ker_idx = 1;
        p_dI_idx = 0; // Store result in 0
    } else {
        // I(0), G(1) -> dK(0)
        num_planes = 2;
        p_img_idx = 0;
        p_grad_idx = 1;
        p_dK_idx = 0; // Store result in 0
    }

    auto complex_opts = grad_output.options().dtype(
            std::is_same<scalar_t, double>::value ? torch::kComplexDouble : torch::kComplexFloat
    );
    auto buffer = torch::empty({B * num_planes, H, W_c}, complex_opts);
    scalar_t* buffer_real_ptr = (scalar_t*)buffer.data_ptr();
    
    int row_stride_real = 2 * W_c;
    int batch_stride_real = H * row_stride_real;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(grad_output.device().index());

    // 1. Pack
    int threads = 256;
    int blocks = (H * W + threads - 1) / threads;

    const scalar_t* img_p = (compute_dK) ? image.contiguous().data_ptr<scalar_t>() : nullptr;
    const scalar_t* ker_p = (compute_dI) ? kernel.contiguous().data_ptr<scalar_t>() : nullptr;
    const scalar_t* grad_p = grad_contig.data_ptr<scalar_t>();

    pack_bwd_inplace_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
        img_p, ker_p, grad_p,
        buffer_real_ptr,
        H, W, h, w, out_h, out_w,
        H*W, broadcast_kernel ? 0 : h*w, out_h*out_w,
        row_stride_real, batch_stride_real,
        p_img_idx, p_ker_idx, p_grad_idx,
        B
    );

    // 2. FFT
    using Traits = FFTTraits<scalar_t>;
    cufftHandle plan_r2c = CuFFTPlanCache::instance().get_plan(H, W, B * num_planes, Traits::R2C_TYPE, true);
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
        p_img_idx, p_ker_idx, p_grad_idx,
        p_dI_idx, p_dK_idx,
        B
    );

    // 4. IFFT
    // How many planes to IFFT?
    // If both: 2 planes (0 and 1).
    // If single: 1 plane (0).
    int ifft_planes = (compute_dI && compute_dK) ? 2 : 1;

    cufftHandle plan_c2r = CuFFTPlanCache::instance().get_plan(H, W, B * ifft_planes, Traits::C2R_TYPE, true);
    CUFFT_CHECK(cufftSetStream(plan_c2r, stream));

    if constexpr (std::is_same<scalar_t, float>::value) {
        CUFFT_CHECK(cufftExecC2R(plan_c2r, (cufftComplex*)buffer.data_ptr(), buffer_real_ptr));
    } else {
        CUFFT_CHECK(cufftExecZ2D(plan_c2r, (cufftDoubleComplex*)buffer.data_ptr(), buffer_real_ptr));
    }

    // 5. Finalize
    torch::Tensor dI, dK;
    if (compute_dI) dI = torch::empty({B, H, W}, image.options());

    // For dK, we allocate B-sized tensor first, then sum if broadcast
    torch::Tensor dK_expanded;
    if (compute_dK) dK_expanded = torch::empty({B, h, w}, image.options());

    blocks = (H * W + threads - 1) / threads;

    finalize_bwd_inplace_kernel<scalar_t><<<dim3(blocks, 1, B), threads, 0, stream>>>(
        buffer_real_ptr,
        compute_dI ? dI.data_ptr<scalar_t>() : nullptr,
        compute_dK ? dK_expanded.data_ptr<scalar_t>() : nullptr,
        H, W, h, w,
        row_stride_real, batch_stride_real,
        H*W, h*w,
        p_dI_idx, p_dK_idx,
        B
    );

    std::vector<torch::Tensor> ret;
    if (compute_dI) ret.push_back(dI.contiguous().reshape(image.sizes()));
    else ret.push_back(torch::Tensor());

    if (compute_dK) {
        if (broadcast_kernel) {
            dK = dK_expanded.sum(0, true).reshape(kernel.sizes());
        } else {
            dK = dK_expanded.contiguous().reshape(kernel.sizes());
        }
        ret.push_back(dK);
    } else {
        ret.push_back(torch::Tensor());
    }

    return ret;
}

std::vector<torch::Tensor> fft_cc_backward_cuda(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel, std::vector<bool> output_mask) {
    return AT_DISPATCH_SWITCH(grad_output.scalar_type(), "fft_cc_backward_cuda",
        AT_DISPATCH_CASE(at::kFloat, [&] { return _fft_cc_backward_cuda_impl<float>(grad_output, image, kernel, output_mask); })
        AT_DISPATCH_CASE(at::kDouble, [&] { return _fft_cc_backward_cuda_impl<double>(grad_output, image, kernel, output_mask); })
    );
}
