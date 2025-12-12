#include <torch/extension.h>
#include "fft_cc.h"

// ==================================================================================
// WRAPPER FUNCTION
// ==================================================================================
torch::Tensor fft_cross_correlation(
    torch::Tensor image,
    torch::Tensor kernel,
    bool normalize = false
) {
    if (image.device().is_cpu()) {
        return fft_cross_correlation_cpu(image, kernel, normalize);
    }
#ifdef WITH_CUDA
    if (image.device().is_cuda()) {
        return fft_cross_correlation_cuda(image, kernel, normalize);
    }
#endif
    TORCH_CHECK(false, "Unsupported device: ", image.device());
}

// ==================================================================================
// CLASS API
// ==================================================================================
class VeryFastNormalizedCrossCorrelation {
public:
    VeryFastNormalizedCrossCorrelation(int img_h, int img_w, int ker_h, int ker_w, bool normalize)
        : img_h_(img_h), img_w_(img_w),
          ker_h_(ker_h), ker_w_(ker_w),
          normalize_(normalize)
    {
#ifdef WITH_CUDA
        // Lazily created or created here?
        // We can create it unconditionally if CUDA is available, or only if we detect CUDA capability.
        // But since constructor doesn't take device, we prepare for CUDA.
        if (torch::cuda::is_available()) {
            cuda_impl_ = create_cuda_impl(img_h, img_w, ker_h, ker_w, normalize);
        }
#endif
    }

    ~VeryFastNormalizedCrossCorrelation() {
#ifdef WITH_CUDA
        if (cuda_impl_) {
            destroy_cuda_impl(cuda_impl_);
        }
#endif
    }

    torch::Tensor forward(torch::Tensor image, torch::Tensor kernel) {
        if (image.device().is_cpu()) {
            return fft_cross_correlation_cpu(image, kernel, normalize_);
        }
#ifdef WITH_CUDA
        if (image.device().is_cuda()) {
            TORCH_CHECK(cuda_impl_ != nullptr, "CUDA implementation not initialized.");
            return forward_cuda_impl(cuda_impl_, image, kernel);
        }
#endif
        TORCH_CHECK(false, "Unsupported device: ", image.device());
    }

private:
    int img_h_, img_w_;
    int ker_h_, ker_w_;
    bool normalize_;

    void* cuda_impl_ = nullptr;
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
