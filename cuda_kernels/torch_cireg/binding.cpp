#include <torch/extension.h>
#include "fft_cc.h"
#include "local_std.h"
#include "fft_zncc.h"

// ==================================================================================
// WRAPPER FUNCTION (Internal - FFT CC)
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
// WRAPPER FUNCTION (Internal - Local Std)
// ==================================================================================
std::vector<torch::Tensor> local_std_forward_impl(torch::Tensor image, torch::Tensor ones, double eps) {
    if (image.device().is_cpu()) {
        return local_std_forward_cpu(image, ones, eps);
    }
#ifdef WITH_CUDA
    if (image.device().is_cuda()) {
        return local_std_forward_cuda(image, ones, eps);
    }
#endif
    TORCH_CHECK(false, "Unsupported device: ", image.device());
}

torch::Tensor local_std_backward_impl(torch::Tensor grad_output, torch::Tensor image, torch::Tensor ones, torch::Tensor std_map, torch::Tensor s1, double eps) {
    if (image.device().is_cpu()) {
        return local_std_backward_cpu(grad_output, image, ones, std_map, s1, eps);
    }
#ifdef WITH_CUDA
    if (image.device().is_cuda()) {
        return local_std_backward_cuda(grad_output, image, ones, std_map, s1, eps);
    }
#endif
    TORCH_CHECK(false, "Unsupported device: ", image.device());
}

// ==================================================================================
// FFT CC AUTOGRAD
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
        
        return {grads[0], grads[1]};
    }
};

torch::Tensor fft_cc(torch::Tensor image, torch::Tensor kernel) {
    return FftCrossCorrelationFunction::apply(image, kernel);
}

// ==================================================================================
// LOCAL STD AUTOGRAD
// ==================================================================================

class LocalStdFunction : public torch::autograd::Function<LocalStdFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor image, torch::Tensor ones, double eps) {
        auto res = local_std_forward_impl(image, ones, eps);
        auto std_map = res[0];
        auto s1 = res[1];
        
        ctx->save_for_backward({image, ones, std_map, s1});
        ctx->saved_data["eps"] = eps; // Save double via map
        
        return std_map;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto image = saved[0];
        auto ones = saved[1];
        auto std_map = saved[2];
        auto s1 = saved[3];
        double eps = ctx->saved_data["eps"].toDouble();
        
        auto grad_output = grad_outputs[0]; // dStd
        
        auto d_image = local_std_backward_impl(grad_output.contiguous(), image, ones, std_map, s1, eps);

        // Return grads: image, ones, eps
        // ones gradient is None (we assume it is constant/fixed/not optimized)
        return {d_image, torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor local_std(torch::Tensor image, torch::Tensor ones, double eps) {
    return LocalStdFunction::apply(image, ones, eps);
}

// ==================================================================================
// FFT ZNCC OPS
// ==================================================================================
std::vector<torch::Tensor> fft_zncc_forward_impl(torch::Tensor image, torch::Tensor kernel, double eps) {
    if (image.device().is_cpu()) {
        return fft_zncc_forward_cpu(image, kernel, eps);
    }
#ifdef WITH_CUDA
    if (image.device().is_cuda()) {
        return fft_zncc_forward_cuda(image, kernel, eps);
    }
#endif
    TORCH_CHECK(false, "Unsupported device: ", image.device());
}

std::vector<torch::Tensor> fft_zncc_backward_impl(torch::Tensor grad_output, torch::Tensor image, torch::Tensor kernel, torch::Tensor std_map, torch::Tensor s1, torch::Tensor k_norm, double eps) {
    if (image.device().is_cpu()) {
        return fft_zncc_backward_cpu(grad_output, image, kernel, std_map, s1, k_norm, eps);
    }
#ifdef WITH_CUDA
    if (image.device().is_cuda()) {
        return fft_zncc_backward_cuda(grad_output, image, kernel, std_map, s1, k_norm, eps);
    }
#endif
    TORCH_CHECK(false, "Unsupported device: ", image.device());
}

class FftZnccFunction : public torch::autograd::Function<FftZnccFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor image, torch::Tensor kernel, double eps) {
        auto res = fft_zncc_forward_impl(image, kernel, eps);
        auto zncc = res[0];
        auto std_map = res[1];
        auto s1 = res[2];
        auto k_norm = res[3];
        
        ctx->save_for_backward({image, kernel, std_map, s1, k_norm});
        ctx->saved_data["eps"] = eps;
        
        return zncc;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto image = saved[0];
        auto kernel = saved[1];
        auto std_map = saved[2];
        auto s1 = saved[3];
        auto k_norm = saved[4];
        double eps = ctx->saved_data["eps"].toDouble();
        
        auto grad_output = grad_outputs[0];
        
        auto grads = fft_zncc_backward_impl(grad_output.contiguous(), image, kernel, std_map, s1, k_norm, eps);
        
        return {grads[0], grads[1], torch::Tensor()};
    }
};

torch::Tensor fft_zncc(torch::Tensor image, torch::Tensor kernel, double eps) {
    return FftZnccFunction::apply(image, kernel, eps);
}

// ==================================================================================
// PYBIND11 MODULE
// ==================================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fft_cc", &fft_cc, "FFT Cross-Correlation (Autograd Support)",
          py::arg("image"), py::arg("kernel"));
          
    m.def("local_std", &local_std, "Local Standard Deviation (Autograd Support)",
          py::arg("image"), py::arg("ones"), py::arg("eps")=1e-5);

    // Raw versions for manual usage
    m.def("fft_cc_forward", &fft_cc_forward_impl, "FFT Cross-Correlation Forward (No Autograd)");
    m.def("fft_cc_backward", &fft_cc_backward_impl, "FFT Cross-Correlation Backward (No Autograd)");
    
    m.def("local_std_forward", &local_std_forward_impl, "Local Std Forward (No Autograd)");
    m.def("local_std_backward", &local_std_backward_impl, "Local Std Backward (No Autograd)");
    
    m.def("fft_zncc", &fft_zncc, "FFT ZNCC (Autograd Support)",
          py::arg("image"), py::arg("kernel"), py::arg("eps")=1e-5);
          
    m.def("fft_zncc_forward", &fft_zncc_forward_impl, "FFT ZNCC Forward (No Autograd)");
    m.def("fft_zncc_backward", &fft_zncc_backward_impl, "FFT ZNCC Backward (No Autograd)");
}
