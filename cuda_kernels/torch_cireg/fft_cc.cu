#include <torch/extension.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuComplex.h>
#include <map>
#include <list> // Added for LRU Cache
#include <mutex>
#include <tuple>
#include <vector>
#include <iostream>
#include <cmath> 

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

using PlanKey = std::tuple<int, int, int, int>;

// --- Helper: Reshape Output ---
std::vector<int64_t> change_h_w_shapes(const torch::Tensor& tensor, const int64_t& H, const int64_t& W) {
    std::vector<int64_t> out;
    out.reserve(tensor.dim()); 
    out.insert(out.end(), tensor.sizes().begin(), tensor.sizes().end() - 2);
    out.insert(out.end(), {H,W});
    return out;
}

// ==================================================================================
// INFRASTRUCTURE: PLAN CACHE (For Functional Interface - Global Singleton)
// ==================================================================================
class CuFFTPlanCache {
public:
    static CuFFTPlanCache& instance() { static CuFFTPlanCache i; return i; }
    
    cufftHandle get_plan(int height, int width, int batch_size, cufftType type) {
        std::lock_guard<std::mutex> lock(mutex_);
        int type_id = (type == CUFFT_R2C) ? 0 : 1; 
        PlanKey key = std::make_tuple(height, width, batch_size, type_id);
        
        if (cache_.find(key) != cache_.end()) return cache_[key];

        cufftHandle plan;
        int n[] = {height, width};
        int width_complex = width / 2 + 1;
        int inembed[2]  = {height, width};
        int onembed[2]  = {height, width_complex};

        int idist = (type == CUFFT_R2C) ? height * width : height * width_complex;
        int odist = (type == CUFFT_R2C) ? height * width_complex : height * width;

        if (type == CUFFT_R2C) {
            CUFFT_CHECK(cufftPlanMany(&plan, 2, n, inembed, 1, idist, onembed, 1, odist, type, batch_size));
        } else {
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

__global__ void pack_image_kernel(
    const float* __restrict__ img_ptr, 
    float* __restrict__ arena_ptr,
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
    
    float px_val = img_ptr[b * img_stride + idx];
    
    arena_ptr[batch_base] = px_val; 
    if (use_zncc) {
        arena_ptr[batch_base + 1 * arena_stride] = px_val * px_val;
    }
}

__global__ void pack_template_kernel(
    const float* __restrict__ ker_ptr,
    float* __restrict__ arena_ptr,
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

    float k_val = ker_ptr[b * ker_stride + idx];

    if (use_zncc) {
        arena_ptr[batch_base + 2 * arena_stride] = k_val;
        arena_ptr[batch_base + 3 * arena_stride] = 1.0f;
    } else {
        arena_ptr[batch_base + 1 * arena_stride] = k_val;
    }
}

__global__ void spectral_math_kernel(
    cuComplex* __restrict__ data_ptr, 
    float scale, 
    int plane_stride, 
    bool use_zncc,
    int batch_offset, int batch_global
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_b = blockIdx.z;
    int b = local_b + batch_offset;

    if (b >= batch_global) return;
    if (idx >= plane_stride) return;

    int stack_depth = use_zncc ? 4 : 2;
    int base = b * stack_depth * plane_stride;

    auto cmul_conj_scale = [&](cuComplex x, cuComplex y) {
        float r = cuCrealf(x)*cuCrealf(y) + cuCimagf(x)*cuCimagf(y);
        float i = cuCimagf(x)*cuCrealf(y) - cuCrealf(x)*cuCimagf(y);
        return make_cuComplex(r*scale, i*scale);
    };

    if (use_zncc) {
        cuComplex I  = data_ptr[base + 0 * plane_stride + idx];
        cuComplex I2 = data_ptr[base + 1 * plane_stride + idx];
        cuComplex T  = data_ptr[base + 2 * plane_stride + idx];
        cuComplex O  = data_ptr[base + 3 * plane_stride + idx];
        
        data_ptr[base + 0 * plane_stride + idx] = cmul_conj_scale(I, T); 
        data_ptr[base + 1 * plane_stride + idx] = cmul_conj_scale(I, O);
        data_ptr[base + 2 * plane_stride + idx] = cmul_conj_scale(I2, O);
    } else {
        cuComplex I = data_ptr[base + 0 * plane_stride + idx];
        cuComplex T = data_ptr[base + 1 * plane_stride + idx];
        data_ptr[base + 0 * plane_stride + idx] = cmul_conj_scale(I, T);
    }
}

__global__ void finalize_kernel(
    const float* __restrict__ arena_ptr,
    float* __restrict__ out_ptr,
    const float* __restrict__ ker_norms, 
    float sqrt_N,                        
    int plane_stride, 
    bool use_zncc,
    int batch_offset, int batch_global
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_b = blockIdx.z;
    int b = local_b + batch_offset;

    if (b >= batch_global) return;
    if (idx >= plane_stride) return;

    int out_base = b * plane_stride;

    if (use_zncc) {
        int base = b * (4 * plane_stride); 
        float num    = arena_ptr[base + 0 * plane_stride + idx];
        float sum_i  = arena_ptr[base + 1 * plane_stride + idx];
        float sum_i2 = arena_ptr[base + 2 * plane_stride + idx];

        float N = sqrt_N * sqrt_N;
        float var_term = N * sum_i2 - (sum_i * sum_i);
        var_term = fmaxf(var_term, 0.0f);
        
        float t_norm = ker_norms[b];

        if (var_term < 1e-5f || t_norm < 1e-6f) {
            out_ptr[out_base + idx] = 0.0f; 
        } else {
            out_ptr[out_base + idx] = (num * sqrt_N) * rsqrtf(var_term) * (1.0f / t_norm);
        }
    } else {
        int base = b * (2 * plane_stride);
        out_ptr[out_base + idx] = arena_ptr[base + idx];
    }
}

// ==================================================================================
// SHARED IMPLEMENTATION
// ==================================================================================
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
    auto cpx_arena  = torch::zeros({batch_size * stack_depth, img_h, width_complex}, 
                                     image.options().dtype(torch::kComplexFloat));

    // 2. Statistics & Pre-processing
    torch::Tensor ker_norms;
    torch::Tensor kernel_to_pack = kernel;
    float sqrt_N = 1.0f;

    if (normalize) {
        auto k_flat = kernel.reshape({batch_size, -1});
        auto k_mean = k_flat.mean(1, true);
        auto k_centered = k_flat - k_mean; 
        
        ker_norms = k_centered.norm(2, {1});
        kernel_to_pack = k_centered.view({batch_size, kernel_h, kernel_w}).contiguous();
        sqrt_N = std::sqrt((float)(kernel_h * kernel_w));
    } else {
        ker_norms = torch::empty({1}, image.options()); 
    }

    // 3. Step 1: Pack
    int plane_size_bytes = batch_size * arena_stride * sizeof(float);
    float* raw_arena = real_arena.data_ptr<float>();

    if (normalize) {
        cudaMemsetAsync(raw_arena + 2 * batch_size * arena_stride, 0, 2 * plane_size_bytes, stream);
    } else {
        cudaMemsetAsync(raw_arena + 1 * batch_size * arena_stride, 0, plane_size_bytes, stream);
    }

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
        pack_image_kernel<<<g, 256, 0, stream>>>(
            image.data_ptr<float>(), real_arena.data_ptr<float>(),
            img_h, img_w, img_stride, arena_stride, normalize, batch_offset, batch_size
        );
    });

    int total_ker_pixels = kernel_h * kernel_w;
    int ker_xblocks = (total_ker_pixels + 255) / 256;
    launch_loop_batches([&](int batch_offset, int batches_here) {
        dim3 g(ker_xblocks, 1, batches_here);
        pack_template_kernel<<<g, 256, 0, stream>>>(
            kernel_to_pack.data_ptr<float>(), real_arena.data_ptr<float>(),
            kernel_h, kernel_w, img_w, ker_stride, arena_stride, normalize, batch_offset, batch_size
        );
    });

    // 4. FFT
    CUFFT_CHECK(cufftSetStream(plan_r2c, stream));
    CUFFT_CHECK(cufftExecR2C(plan_r2c, real_arena.data_ptr<float>(), reinterpret_cast<cufftComplex*>(cpx_arena.data_ptr())));

    // 5. Spectral Math
    int total_spec = img_h * width_complex;
    float fft_scale = 1.0f / (float)total_pixels;
    int spec_xblocks = (total_spec + 255) / 256;
    launch_loop_batches([&](int batch_offset, int batches_here) {
        dim3 g(spec_xblocks, 1, batches_here);
        spectral_math_kernel<<<g, 256, 0, stream>>>(
            reinterpret_cast<cuComplex*>(cpx_arena.data_ptr()), 
            fft_scale, total_spec, normalize, batch_offset, batch_size
        );
    });

    // 6. IFFT
    CUFFT_CHECK(cufftSetStream(plan_c2r, stream));
    CUFFT_CHECK(cufftExecC2R(plan_c2r, reinterpret_cast<cufftComplex*>(cpx_arena.data_ptr()), real_arena.data_ptr<float>()));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 7. Finalize
    auto full_result = torch::empty({batch_size, img_h, img_w}, image.options());
    int finalize_xblocks = (total_pixels + 255) / 256;
    launch_loop_batches([&](int batch_offset, int batches_here) {
        dim3 g(finalize_xblocks, 1, batches_here);
        finalize_kernel<<<g, 256, 0, stream>>>(
            real_arena.data_ptr<float>(),
            full_result.data_ptr<float>(),
            ker_norms.data_ptr<float>(), 
            sqrt_N,
            total_pixels, normalize, batch_offset, batch_size
        );
    });

    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // 8. Crop
    const auto crop_h = img_h - kernel_h + 1;
    const auto crop_w = img_w - kernel_w + 1;

    return full_result.slice(1, 0, crop_h).slice(2, 0,crop_w)
                .reshape(change_h_w_shapes(image, crop_h, crop_w));
}

// ==================================================================================
// 1. FUNCTIONAL API
// ==================================================================================
torch::Tensor fft_cross_correlation(
    torch::Tensor image, 
    torch::Tensor kernel, 
    bool normalize = false
) {
    int img_h = image.size(-2);
    int img_w = image.size(-1);
    int batch_size = image.numel() / (img_h * img_w);
    int stack_depth = normalize ? 4 : 2;

    cufftHandle plan_r2c = CuFFTPlanCache::instance().get_plan(img_h, img_w, batch_size * stack_depth, CUFFT_R2C);
    cufftHandle plan_c2r = CuFFTPlanCache::instance().get_plan(img_h, img_w, batch_size * stack_depth, CUFFT_C2R);

    return fft_cross_correlation_impl(image, kernel, normalize, plan_r2c, plan_c2r);
}

// ==================================================================================
// 2. CLASS API (Dynamic Batch Handling with LRU Cache of 2)
// ==================================================================================

struct ClassPlanPair {
    cufftHandle r2c;
    cufftHandle c2r;
};

class VeryFastNormalizedCrossCorrelation {
public:
    // Init: Capture fixed dimensions
    VeryFastNormalizedCrossCorrelation(int img_h, int img_w, int ker_h, int ker_w, bool normalize) 
        : img_h_(img_h), img_w_(img_w), 
          ker_h_(ker_h), ker_w_(ker_w), 
          normalize_(normalize) 
    {}

    // Cleanup: Destroy all plans in cache
    ~VeryFastNormalizedCrossCorrelation() {
        for (auto& entry : plan_cache_) {
            cufftDestroy(entry.second.r2c);
            cufftDestroy(entry.second.c2r);
        }
        plan_cache_.clear();
    }

    torch::Tensor forward(torch::Tensor image, torch::Tensor kernel) {
        TORCH_CHECK(image.is_cuda() && kernel.is_cuda(), "Inputs must be on CUDA.");
        TORCH_CHECK(image.size(-2) == img_h_ && image.size(-1) == img_w_, 
                    "Image dims mismatch class init. Expected ", img_h_, "x", img_w_);
        TORCH_CHECK(kernel.size(-2) == ker_h_ && kernel.size(-1) == ker_w_, 
                    "Kernel dims mismatch class init. Expected ", ker_h_, "x", ker_w_);

        int current_batch_size = image.numel() / (img_h_ * img_w_);

        // Get plans (Cached or New)
        ClassPlanPair plans = get_or_create_plans(current_batch_size);

        return fft_cross_correlation_impl(
            image, kernel, normalize_, 
            plans.r2c, plans.c2r, 
            current_batch_size
        );
    }

private:
    int img_h_, img_w_;
    int ker_h_, ker_w_;
    bool normalize_;
    
    // LRU Cache Implementation:
    // List stores pairs of (batch_size, Plans). 
    // Front is MRU (Most Recently Used), Back is LRU.
    std::list<std::pair<int, ClassPlanPair>> plan_cache_;
    std::mutex cache_mutex_;

    ClassPlanPair get_or_create_plans(int batch_size) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        // 1. Search in List (Linear scan is fast for size 2)
        for (auto it = plan_cache_.begin(); it != plan_cache_.end(); ++it) {
            if (it->first == batch_size) {
                // Hit! Move to Front (MRU)
                plan_cache_.splice(plan_cache_.begin(), plan_cache_, it);
                return it->second;
            }
        }

        // 2. Miss! Create New Plans
        ClassPlanPair new_plans;
        
        int n[] = {img_h_, img_w_};
        int width_complex = img_w_ / 2 + 1;
        int stack_depth = normalize_ ? 4 : 2;
        int total_layers = batch_size * stack_depth;

        int inembed_r2c[2]  = {img_h_, img_w_};
        int onembed_r2c[2]  = {img_h_, width_complex};
        int idist_r2c = img_h_ * img_w_;
        int odist_r2c = img_h_ * width_complex;

        int onembed_c2r[2]  = {img_h_, img_w_};
        int inembed_c2r[2]  = {img_h_, width_complex};
        int idist_c2r = img_h_ * width_complex;
        int odist_c2r = img_h_ * img_w_;

        CUFFT_CHECK(cufftPlanMany(&new_plans.r2c, 2, n, 
                                  inembed_r2c, 1, idist_r2c, 
                                  onembed_r2c, 1, odist_r2c, 
                                  CUFFT_R2C, total_layers));

        CUFFT_CHECK(cufftPlanMany(&new_plans.c2r, 2, n, 
                                  inembed_c2r, 1, idist_c2r, 
                                  onembed_c2r, 1, odist_c2r, 
                                  CUFFT_C2R, total_layers));

        // 3. Insert at Front
        plan_cache_.push_front({batch_size, new_plans});

        // 4. Evict LRU if size > 2
        if (plan_cache_.size() > 2) {
            auto& last_entry = plan_cache_.back();
            cufftDestroy(last_entry.second.r2c);
            cufftDestroy(last_entry.second.c2r);
            plan_cache_.pop_back();
        }

        return new_plans;
    }
};

// ==================================================================================
// PYBIND11 MODULE
// ==================================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fft_cross_correlation", &fft_cross_correlation, 
          "High-Performance FFT Cross-Correlation / ZNCC",
          py::arg("image"), py::arg("kernel"), py::arg("normalize") = false);

    py::class_<VeryFastNormalizedCrossCorrelation>(m, "VeryFastNormalizedCrossCorrelation")
        .def(py::init<int, int, int, int, bool>(), 
             py::arg("img_h"), py::arg("img_w"), 
             py::arg("ker_h"), py::arg("ker_w"), py::arg("normalize")=true)
        .def("forward", &VeryFastNormalizedCrossCorrelation::forward, 
             py::arg("image"), py::arg("kernel"));
}