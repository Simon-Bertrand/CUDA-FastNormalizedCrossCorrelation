#include <torch/extension.h>
#include "fft_cc.h"

// ==================================================================================
// WRAPPER FUNCTION
// ==================================================================================
torch::Tensor fft_cc_forward(torch::Tensor image, torch::Tensor kernel) {
    if (image.device().is_cpu()) {
        return fft_cc_forward_cpu(image, kernel);
    }
#ifdef WITH_CUDA
    if (image.device().is_cuda()) {
        return fft_cc_forward_cuda(image, kernel);
    }
#endif
    TORCH_CHECK(false, "Unsupported device: ", image.device());
}

std::vector<torch::Tensor> fft_cc_backward(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel) {
    if (image.device().is_cpu()) {
        return fft_cc_backward_cpu(grad_output, image, kernel);
    }
#ifdef WITH_CUDA
    if (image.device().is_cuda()) {
        return fft_cc_backward_cuda(grad_output, image, kernel);
    }
#endif
    TORCH_CHECK(false, "Unsupported device: ", image.device());
}

// ==================================================================================
// PYBIND11 MODULE
// ==================================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fft_cc_forward", &fft_cc_forward, "FFT Cross-Correlation Forward",
          py::arg("image"), py::arg("kernel"));

    m.def("fft_cc_backward", &fft_cc_backward, "FFT Cross-Correlation Backward",
          py::arg("grad_output"), py::arg("image"), py::arg("kernel"));
}
