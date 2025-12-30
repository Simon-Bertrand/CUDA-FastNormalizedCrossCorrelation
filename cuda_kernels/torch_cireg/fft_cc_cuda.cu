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
// MACROS & ERROR HANDLING
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

// Tuple: (height, width, batch_size, type_id)
// type_id distinguishes between R2C vs D2Z (for float vs double)
using PlanKey = std::tuple<int, int, int, int>;

// Note: change_h_w_shapes is now inline in fft_cc.h

// ==================================================================================
// TRAITS FOR FLOAT VS DOUBLE
// ==================================================================================
template <typename T> struct FFTTraits;

template <> struct FFTTraits<float> {
    using ComplexType = cufftComplex;
    using RealType = float;
    static const cufftType R2C_TYPE = CUFFT_R2C;
    static const cufftType C2R_TYPE = CUFFT_C2R;

    __device__ static ComplexType make_complex(float r, float i) {
        return make_cuComplex(r, i);
    }
    __device__ static float real(ComplexType c) { return cuCrealf(c); }
    __device__ static float imag(ComplexType c) { return cuCimagf(c); }
    __device__ static float rsqrt(float x) { return rsqrtf(x); }
    __device__ static float fmax(float x, float y) { return fmaxf(x, y); }
    __device__ static float fmin(float x, float y) { return fminf(x, y); }
    __device__ static bool isnan(float x) { return isnan(x); }
};

template <> struct FFTTraits<double> {
    using ComplexType = cufftDoubleComplex;
    using RealType = double;
    static const cufftType R2C_TYPE = CUFFT_D2Z;
    static const cufftType C2R_TYPE = CUFFT_Z2D;

    __device__ static ComplexType make_complex(double r, double i) {
        return make_cuDoubleComplex(r, i);
    }
    __device__ static double real(ComplexType c) { return cuCreal(c); }
    __device__ static double imag(ComplexType c) { return cuCimag(c); }
    __device__ static double rsqrt(double x) { return 1.0 / sqrt(x); }
    __device__ static double fmax(double x, double y) { return fmax(x, y); }
    __device__ static double fmin(double x, double y) { return fmin(x, y); }
    __device__ static bool isnan(double x) { return isnan(x); }
};

// ==================================================================================
// INFRASTRUCTURE: PLAN CACHE
// ==================================================================================
class CuFFTPlanCache {
public:
    static CuFFTPlanCache& instance() { static CuFFTPlanCache i; return i; }
    
    cufftHandle get_plan(int height, int width, int batch_size, cufftType type) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Map cufftType to an integer ID for the key
        // We can just use the enum value directly as int
        int type_id = (int)type;
        PlanKey key = std::make_tuple(height, width, batch_size, type_id);
        
        if (cache_.find(key) != cache_.end()) return cache_[key];

        cufftHandle plan;
        int n[] = {height, width};
        int width_complex = width / 2 + 1;
        int inembed[2]  = {height, width};
        int onembed[2]  = {height, width_complex};

        // Determine dists based on type
        bool is_r2c = (type == CUFFT_R2C || type == CUFFT_D2Z);

        int idist, odist;
        if (is_r2c) {
            idist = height * width;
            odist = height * width_complex;
            CUFFT_CHECK(cufftPlanMany(&plan, 2, n, inembed, 1, idist, onembed, 1, odist, type, batch_size));
        } else {
            idist = height * width_complex;
            odist = height * width;
            // For C2R/Z2D, input is complex, output is real
            CUFFT_CHECK(cufftPlanMany(&plan, 2, n, onembed, 1, idist, inembed, 1, odist, type, batch_size));
        }

        cache_[key] = plan;
        return plan;
    }
private:
    std::map<PlanKey, cufftHandle> cache_;
    std::mutex mutex_;
};

// ==================================================================================
// CUDA KERNELS
// ==================================================================================

template <typename scalar_t>
__global__ void pack_image_kernel(
    const scalar_t* __restrict__ img_ptr,
    scalar_t* __restrict__ arena_ptr,
    const scalar_t* __restrict__ img_means, // [B] or nullptr
    int img_h, int img_w, 
    int img_stride, int arena_stride,
    bool use_zncc,
    int batch_offset, int batch_global
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = img_h * img_w;
    int local_b = blockIdx.z;       
    int b = local_b + batch_offset;   

    if (b >= batch_global) return;    
    if (idx >= total_pixels) return;

    int stack_depth = use_zncc ? 4 : 2;
    int batch_base = b * (stack_depth * arena_stride) + idx;
    
    scalar_t px_val = img_ptr[b * img_stride + idx];

    if (use_zncc && img_means != nullptr) {
        // Subtract mean
        px_val -= img_means[b];
    }
    
    arena_ptr[batch_base] = px_val; 
    if (use_zncc) {
        arena_ptr[batch_base + 1 * arena_stride] = px_val * px_val;
    }
}

template <typename scalar_t>
__global__ void pack_template_kernel(
    const scalar_t* __restrict__ ker_ptr,
    scalar_t* __restrict__ arena_ptr,
    int kernel_h, int kernel_w,
    int img_w, 
    int ker_stride, int arena_stride,
    bool use_zncc,
    int batch_offset, int batch_global
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_kernel_pixels = kernel_h * kernel_w;
    int local_b = blockIdx.z;       
    int b = local_b + batch_offset;   

    if (b >= batch_global) return;    
    if (idx >= total_kernel_pixels) return;

    int k_x = idx % kernel_w;
    int k_y = idx / kernel_w;
    int arena_idx = k_y * img_w + k_x;

    int stack_depth = use_zncc ? 4 : 2;
    int batch_base = b * (stack_depth * arena_stride) + arena_idx;

    scalar_t k_val = ker_ptr[b * ker_stride + idx];

    if (use_zncc) {
        arena_ptr[batch_base + 2 * arena_stride] = k_val;
        arena_ptr[batch_base + 3 * arena_stride] = static_cast<scalar_t>(1.0);
    } else {
        arena_ptr[batch_base + 1 * arena_stride] = k_val;
    }
}

template <typename scalar_t>
__global__ void spectral_math_kernel(
    typename FFTTraits<scalar_t>::ComplexType* __restrict__ data_ptr,
    scalar_t scale,
    int plane_stride, 
    bool use_zncc,
    int batch_offset, int batch_global
) {
    using ComplexT = typename FFTTraits<scalar_t>::ComplexType;
    using Traits = FFTTraits<scalar_t>;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_b = blockIdx.z;
    int b = local_b + batch_offset;

    if (b >= batch_global) return;
    if (idx >= plane_stride) return;

    int stack_depth = use_zncc ? 4 : 2;
    int base = b * stack_depth * plane_stride;

    auto cmul_conj_scale = [&](ComplexT x, ComplexT y) {
        scalar_t xr = Traits::real(x);
        scalar_t xi = Traits::imag(x);
        scalar_t yr = Traits::real(y);
        scalar_t yi = Traits::imag(y);

        scalar_t r = xr*yr + xi*yi;
        scalar_t i = xi*yr - xr*yi;
        return Traits::make_complex(r*scale, i*scale);
    };

    if (use_zncc) {
        ComplexT I  = data_ptr[base + 0 * plane_stride + idx];
        ComplexT I2 = data_ptr[base + 1 * plane_stride + idx];
        ComplexT T  = data_ptr[base + 2 * plane_stride + idx];
        ComplexT O  = data_ptr[base + 3 * plane_stride + idx];
        
        // Output P0 (Num) = I * conj(T)
        // Output P1 (Sum_I) = I * conj(O)
        // Output P2 (Sum_I2) = I2 * conj(O)
        data_ptr[base + 0 * plane_stride + idx] = cmul_conj_scale(I, T); 
        data_ptr[base + 1 * plane_stride + idx] = cmul_conj_scale(I, O);
        data_ptr[base + 2 * plane_stride + idx] = cmul_conj_scale(I2, O);
    } else {
        ComplexT I = data_ptr[base + 0 * plane_stride + idx];
        ComplexT T = data_ptr[base + 1 * plane_stride + idx];
        data_ptr[base + 0 * plane_stride + idx] = cmul_conj_scale(I, T);
    }
}

template <typename scalar_t>
__global__ void finalize_kernel(
    const scalar_t* __restrict__ arena_ptr,
    scalar_t* __restrict__ out_ptr,
    const scalar_t* __restrict__ ker_norms,
    scalar_t sqrt_N,
    int arena_stride,   // img_h * img_w
    int img_w,          // to calc y
    int out_w,          // to calc x/y from idx
    int total_out_pixels, // crop_h * crop_w
    bool use_zncc,
    int batch_offset, int batch_global
) {
    using Traits = FFTTraits<scalar_t>;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_b = blockIdx.z;
    int b = local_b + batch_offset;

    if (b >= batch_global) return;
    if (idx >= total_out_pixels) return;

    // Coordinate mapping for Fused Crop
    int out_y = idx / out_w;
    int out_x = idx % out_w;
    int arena_idx = out_y * img_w + out_x;

    int out_base = b * total_out_pixels;

    if (use_zncc) {
        int base = b * (4 * arena_stride); 
        scalar_t num    = arena_ptr[base + 0 * arena_stride + arena_idx];
        scalar_t sum_i  = arena_ptr[base + 1 * arena_stride + arena_idx];
        scalar_t sum_i2 = arena_ptr[base + 2 * arena_stride + arena_idx];

        scalar_t N = sqrt_N * sqrt_N;
        scalar_t var_term = N * sum_i2 - (sum_i * sum_i);
        var_term = Traits::fmax(var_term, static_cast<scalar_t>(0.0));
        
        scalar_t t_norm = ker_norms[b];

        // Use smaller epsilon for double precision to avoid suppressing valid correlations
        scalar_t epsilon = std::is_same<scalar_t, double>::value ? static_cast<scalar_t>(1e-6) : static_cast<scalar_t>(1e-5);

        if (var_term < epsilon || 
            t_norm < epsilon || 
            Traits::isnan(var_term) || 
            Traits::isnan(t_norm) || 
            Traits::isnan(num)) {
            out_ptr[out_base + idx] = static_cast<scalar_t>(0.0);
        } else {
            scalar_t res = (num * sqrt_N) * Traits::rsqrt(var_term) * (static_cast<scalar_t>(1.0) / t_norm);
            res = Traits::fmin(res, static_cast<scalar_t>(1.0));
            res = Traits::fmax(res, static_cast<scalar_t>(-1.0));
            out_ptr[out_base + idx] = res;
        }
    } else {
        int base = b * (2 * arena_stride);
        out_ptr[out_base + idx] = arena_ptr[base + arena_idx];
    }
}

// ==================================================================================
// SHARED IMPLEMENTATION TEMPLATE
// ==================================================================================
template <typename scalar_t>
torch::Tensor fft_cross_correlation_impl(
    torch::Tensor image, 
    torch::Tensor kernel, 
    bool normalize,
    cufftHandle plan_r2c,
    cufftHandle plan_c2r,
    int expected_batch_size = -1 
) {
    TORCH_CHECK(image.is_cuda() && kernel.is_cuda(), "Inputs must be on CUDA.");
    TORCH_CHECK(image.is_contiguous() && kernel.is_contiguous(), "Inputs must be contiguous.");

    using Traits = FFTTraits<scalar_t>;
    using ComplexT = typename Traits::ComplexType;

    int img_h = image.size(-2);
    int img_w = image.size(-1);
    int batch_size = image.numel() / (img_h * img_w);
    int kernel_h = kernel.size(-2);
    int kernel_w = kernel.size(-1);
    int ker_stride = kernel_h * kernel_w;
    int img_stride = img_h * img_w;
    int width_complex = img_w / 2 + 1;

    if (expected_batch_size != -1) {
        TORCH_CHECK(batch_size == expected_batch_size, 
            "Batch size mismatch. Expected ", expected_batch_size, " got ", batch_size);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(image.device().index());

    // 1. Allocation
    int stack_depth = normalize ? 4 : 2;
    int arena_stride = img_h * img_w; 
    auto real_arena = torch::zeros({batch_size * stack_depth, img_h, img_w}, image.options());

    // For Complex tensor allocation, we need to pick correct ComplexFloat/ComplexDouble
    auto complex_options = image.options().dtype(
        std::is_same<scalar_t, double>::value ? torch::kComplexDouble : torch::kComplexFloat
    );
    auto cpx_arena  = torch::empty({batch_size * stack_depth, img_h, width_complex}, complex_options);

    // 2. Statistics & Pre-processing
    torch::Tensor ker_norms;
    torch::Tensor kernel_to_pack = kernel;
    scalar_t sqrt_N = 1.0;

    // Per-image mean for stability
    torch::Tensor img_means;

    if (normalize) {
        auto k_flat = kernel.reshape({batch_size, -1});
        auto k_mean = k_flat.mean(1, true);
        auto k_centered = k_flat - k_mean; 
        
        ker_norms = k_centered.norm(2, {1});
        kernel_to_pack = k_centered.view({batch_size, kernel_h, kernel_w}).contiguous();
        sqrt_N = std::sqrt(static_cast<scalar_t>(kernel_h * kernel_w));

        // Compute image means [B]
        auto img_flat = image.reshape({batch_size, -1});
        img_means = img_flat.mean(1);
    } else {
        ker_norms = torch::empty({1}, image.options()); 
    }

    // 3. Step 1: Pack
    int plane_size_bytes = batch_size * arena_stride * sizeof(scalar_t);
    scalar_t* raw_arena = real_arena.data_ptr<scalar_t>();

    // Note: real_arena is allocated with torch::zeros, so we don't need cudaMemsetAsync.

    const int CUDA_MAX_GRID_Z = 65535;
    auto launch_loop_batches = [&](auto kernel_launcher) {
        int total_batches = batch_size;
        int start = 0;
        while (start < total_batches) {
            int this_batch = std::min(total_batches - start, CUDA_MAX_GRID_Z);
            kernel_launcher(start, this_batch);
            start += this_batch;
        }
    };

    int total_pixels = img_h * img_w;
    int img_xblocks = (total_pixels + 255) / 256;
    launch_loop_batches([&](int batch_offset, int batches_here) {
        dim3 g(img_xblocks, 1, batches_here);
        pack_image_kernel<scalar_t><<<g, 256, 0, stream>>>(
            image.data_ptr<scalar_t>(), real_arena.data_ptr<scalar_t>(),
            normalize ? img_means.data_ptr<scalar_t>() : nullptr,
            img_h, img_w, img_stride, arena_stride, normalize, batch_offset, batch_size
        );
    });

    int total_ker_pixels = kernel_h * kernel_w;
    int ker_xblocks = (total_ker_pixels + 255) / 256;
    launch_loop_batches([&](int batch_offset, int batches_here) {
        dim3 g(ker_xblocks, 1, batches_here);
        pack_template_kernel<scalar_t><<<g, 256, 0, stream>>>(
            kernel_to_pack.data_ptr<scalar_t>(), real_arena.data_ptr<scalar_t>(),
            kernel_h, kernel_w, img_w, ker_stride, arena_stride, normalize, batch_offset, batch_size
        );
    });

    // 4. FFT
    CUFFT_CHECK(cufftSetStream(plan_r2c, stream));

    // Dispatch to correct Exec function
    if constexpr (std::is_same<scalar_t, float>::value) {
        CUFFT_CHECK(cufftExecR2C(plan_r2c, real_arena.data_ptr<scalar_t>(), reinterpret_cast<cufftComplex*>(cpx_arena.data_ptr())));
    } else {
        CUFFT_CHECK(cufftExecD2Z(plan_r2c, real_arena.data_ptr<scalar_t>(), reinterpret_cast<cufftDoubleComplex*>(cpx_arena.data_ptr())));
    }

    // 5. Spectral Math
    int total_spec = img_h * width_complex;
    scalar_t fft_scale = static_cast<scalar_t>(1.0) / static_cast<scalar_t>(total_pixels);
    int spec_xblocks = (total_spec + 255) / 256;
    launch_loop_batches([&](int batch_offset, int batches_here) {
        dim3 g(spec_xblocks, 1, batches_here);
        spectral_math_kernel<scalar_t><<<g, 256, 0, stream>>>(
            reinterpret_cast<ComplexT*>(cpx_arena.data_ptr()),
            fft_scale, total_spec, normalize, batch_offset, batch_size
        );
    });

    // 6. IFFT
    CUFFT_CHECK(cufftSetStream(plan_c2r, stream));
    if constexpr (std::is_same<scalar_t, float>::value) {
        CUFFT_CHECK(cufftExecC2R(plan_c2r, reinterpret_cast<cufftComplex*>(cpx_arena.data_ptr()), real_arena.data_ptr<scalar_t>()));
    } else {
        CUFFT_CHECK(cufftExecZ2D(plan_c2r, reinterpret_cast<cufftDoubleComplex*>(cpx_arena.data_ptr()), real_arena.data_ptr<scalar_t>()));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 7. Finalize & Crop (Fused)
    const auto crop_h = img_h - kernel_h + 1;
    const auto crop_w = img_w - kernel_w + 1;
    int total_out_pixels = crop_h * crop_w;
    auto final_result = torch::empty({batch_size, crop_h, crop_w}, image.options());
    
    int finalize_xblocks = (total_out_pixels + 255) / 256;
    launch_loop_batches([&](int batch_offset, int batches_here) {
        dim3 g(finalize_xblocks, 1, batches_here);
        finalize_kernel<scalar_t><<<g, 256, 0, stream>>>(
            real_arena.data_ptr<scalar_t>(),
            final_result.data_ptr<scalar_t>(),
            ker_norms.data_ptr<scalar_t>(),
            sqrt_N,
            total_pixels, img_w, crop_w, total_out_pixels,
            normalize, batch_offset, batch_size
        );
    });

    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    return final_result.reshape(change_h_w_shapes(image, crop_h, crop_w));
}

// ==================================================================================
// 1. FUNCTIONAL API (CUDA)
// ==================================================================================
torch::Tensor fft_cross_correlation_cuda(
    torch::Tensor image, 
    torch::Tensor kernel, 
    bool normalize
) {
    return AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "fft_cross_correlation_cuda", [&] {
        int img_h = image.size(-2);
        int img_w = image.size(-1);
        int batch_size = image.numel() / (img_h * img_w);
        int stack_depth = normalize ? 4 : 2;

        cufftType r2c_type = FFTTraits<scalar_t>::R2C_TYPE;
        cufftType c2r_type = FFTTraits<scalar_t>::C2R_TYPE;

        cufftHandle plan_r2c = CuFFTPlanCache::instance().get_plan(img_h, img_w, batch_size * stack_depth, r2c_type);
        cufftHandle plan_c2r = CuFFTPlanCache::instance().get_plan(img_h, img_w, batch_size * stack_depth, c2r_type);

        return fft_cross_correlation_impl<scalar_t>(image, kernel, normalize, plan_r2c, plan_c2r);
    });
}

// ==================================================================================
// 2. CLASS API HELPERS
// ==================================================================================

struct ClassPlanPair {
    cufftHandle r2c;
    cufftHandle c2r;
};

class CudaImpl {
public:
    CudaImpl(int img_h, int img_w, int ker_h, int ker_w, bool normalize)
        : img_h_(img_h), img_w_(img_w), 
          ker_h_(ker_h), ker_w_(ker_w), 
          normalize_(normalize) 
    {}

    ~CudaImpl() {
        for (auto& entry : plan_cache_) {
            cufftDestroy(entry.second.r2c);
            cufftDestroy(entry.second.c2r);
        }
        plan_cache_.clear();
    }

    torch::Tensor forward(torch::Tensor image, torch::Tensor kernel) {
        return AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "CudaImpl::forward", [&] {
            int current_batch_size = image.numel() / (img_h_ * img_w_);

            cufftType r2c_type = FFTTraits<scalar_t>::R2C_TYPE;
            cufftType c2r_type = FFTTraits<scalar_t>::C2R_TYPE;

            ClassPlanPair plans = get_or_create_plans(current_batch_size, r2c_type, c2r_type);

            return fft_cross_correlation_impl<scalar_t>(
                image, kernel, normalize_,
                plans.r2c, plans.c2r,
                current_batch_size
            );
        });
    }

private:
    int img_h_, img_w_;
    int ker_h_, ker_w_;
    bool normalize_;
    
    // Key now needs to include type info because plans differ by type.
    // Use pair<BatchSize, TypeId> as key. TypeId can be the r2c cufftType enum value.
    using CacheKey = std::pair<int, int>;
    std::list<std::pair<CacheKey, ClassPlanPair>> plan_cache_;
    std::mutex cache_mutex_;

    ClassPlanPair get_or_create_plans(int batch_size, cufftType r2c_type, cufftType c2r_type) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        CacheKey key = {batch_size, (int)r2c_type};

        for (auto it = plan_cache_.begin(); it != plan_cache_.end(); ++it) {
            if (it->first == key) {
                plan_cache_.splice(plan_cache_.begin(), plan_cache_, it);
                return it->second;
            }
        }

        ClassPlanPair new_plans;
        
        int n[] = {img_h_, img_w_};
        int width_complex = img_w_ / 2 + 1;
        int stack_depth = normalize_ ? 4 : 2;
        int total_layers = batch_size * stack_depth;

        int inembed_r2c[2]  = {img_h_, img_w_};
        int onembed_r2c[2]  = {img_h_, width_complex};

        // idist/odist depend on type size, but cufftPlanMany logic for idist/odist counts ELEMENTS.
        // For R2C: input is H*W real elements. output is H*Wc complex elements.
        int idist_r2c = img_h_ * img_w_;
        int odist_r2c = img_h_ * width_complex;

        // For C2R: input is H*Wc complex elements. output is H*W real elements.
        int onembed_c2r[2]  = {img_h_, img_w_};
        int inembed_c2r[2]  = {img_h_, width_complex};
        int idist_c2r = img_h_ * width_complex;
        int odist_c2r = img_h_ * img_w_;

        CUFFT_CHECK(cufftPlanMany(&new_plans.r2c, 2, n, 
                                  inembed_r2c, 1, idist_r2c, 
                                  onembed_r2c, 1, odist_r2c, 
                                  r2c_type, total_layers));

        CUFFT_CHECK(cufftPlanMany(&new_plans.c2r, 2, n, 
                                  inembed_c2r, 1, idist_c2r, 
                                  onembed_c2r, 1, odist_c2r, 
                                  c2r_type, total_layers));

        plan_cache_.push_front({key, new_plans});

        if (plan_cache_.size() > 4) { // Increase cache size slightly to accommodate float/double switching if happens
            auto& last_entry = plan_cache_.back();
            cufftDestroy(last_entry.second.r2c);
            cufftDestroy(last_entry.second.c2r);
            plan_cache_.pop_back();
        }

        return new_plans;
    }
};

// C-style interface for the opaque pointer interaction
void* create_cuda_impl(int img_h, int img_w, int ker_h, int ker_w, bool normalize) {
    return new CudaImpl(img_h, img_w, ker_h, ker_w, normalize);
}

void destroy_cuda_impl(void* impl) {
    if (impl) delete static_cast<CudaImpl*>(impl);
}

torch::Tensor forward_cuda_impl(void* impl, torch::Tensor image, torch::Tensor kernel) {
    TORCH_CHECK(impl != nullptr, "CUDA impl is null");
    return static_cast<CudaImpl*>(impl)->forward(image, kernel);
}
