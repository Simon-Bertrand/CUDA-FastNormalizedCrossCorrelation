#include <torch/extension.h>
#include "fft_cc.h"

// ==================================================================================
// WRAPPER FUNCTION (Internal)
// ==================================================================================
torch::Tensor fft_cc_forward_impl(torch::Tensor image, torch::Tensor kernel) {
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

std::vector<torch::Tensor> fft_cc_backward_impl(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel) {
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
// AUTOGRAD FUNCTION
// ==================================================================================

class FftCrossCorrelationFunction : public torch::autograd::Function<FftCrossCorrelationFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor image, torch::Tensor kernel) {
        ctx->save_for_backward({image, kernel});
        return fft_cc_forward_impl(image, kernel);
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto image = saved[0];
        auto kernel = saved[1];
        auto grad_output = grad_outputs[0];

        auto grads = fft_cc_backward_impl(grad_output.contiguous(), image, kernel);
        
        // Return gradients: {d_image, d_kernel}
        // Since forward takes (image, kernel), backward returns list of size 2.
        return {grads[0], grads[1]};
    }
};

torch::Tensor fft_cc(torch::Tensor image, torch::Tensor kernel) {
    return FftCrossCorrelationFunction::apply(image, kernel);
}

// ==================================================================================
// PYBIND11 MODULE
// ==================================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fft_cc", &fft_cc, "FFT Cross-Correlation (Autograd Support)",
          py::arg("image"), py::arg("kernel"));
          
    // Keep raw versions for debugging/manual usage if needed, or remove if strict replacement
    m.def("fft_cc_forward", &fft_cc_forward_impl, "FFT Cross-Correlation Forward (No Autograd)");
    m.def("fft_cc_backward", &fft_cc_backward_impl, "FFT Cross-Correlation Backward (No Autograd)");
}
