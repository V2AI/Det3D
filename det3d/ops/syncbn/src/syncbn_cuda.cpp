#include <torch/torch.h>

#include <vector>

at::Tensor batch_norm_transform_input_cuda(
    const at::Tensor input, 
    const at::Tensor gamma,
    const at::Tensor beta,
    const at::Tensor ex,
    const at::Tensor exs,
    float eps,
    float cf);

std::vector<at::Tensor> batch_norm_collect_grad_statistics_cuda(
    const at::Tensor input,
    const at::Tensor gradoutput,
    const at::Tensor gamma,
    const at::Tensor ex,
    const at::Tensor exs,
    float eps,
    float cf);

at::Tensor batch_norm_input_backward_cuda(
    const at::Tensor input,
    const at::Tensor gradoutput,
    const at::Tensor gamma,
    const at::Tensor ex,
    const at::Tensor exs,
    const at::Tensor gradex,
    const at::Tensor gradexs,
    float eps,
    float cf);

std::vector<at::Tensor> batch_norm_collect_statistics_cuda(
    const at::Tensor input);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor batch_norm_transform_input(
    const at::Tensor input,
    const at::Tensor gamma,
    const at::Tensor beta,
    const at::Tensor ex,
    const at::Tensor exs,
    float eps,
    float cf) {
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    CHECK_INPUT(ex);
    CHECK_INPUT(exs);

    return batch_norm_transform_input_cuda(input, gamma, beta, ex, exs, eps, cf);
}

std::vector<at::Tensor> batch_norm_collect_grad_statistics(
    const at::Tensor input,
    const at::Tensor gradoutput,
    const at::Tensor gamma,
    const at::Tensor ex,
    const at::Tensor exs,
    float eps,
    float cf) {
    CHECK_INPUT(input);
    CHECK_INPUT(gradoutput);
    CHECK_INPUT(gamma);
    CHECK_INPUT(ex);
    CHECK_INPUT(exs);

    return batch_norm_collect_grad_statistics_cuda(input, gradoutput, gamma, ex, exs, eps, cf);
}

at::Tensor batch_norm_input_backward(
    const at::Tensor input,
    const at::Tensor gradoutput,
    const at::Tensor gamma,
    const at::Tensor ex,
    const at::Tensor exs,
    const at::Tensor gradex,
    const at::Tensor gradexs,
    float eps,
    float cf) {
    CHECK_INPUT(input);
    CHECK_INPUT(gradoutput);
    CHECK_INPUT(gamma);
    CHECK_INPUT(ex);
    CHECK_INPUT(exs);
    CHECK_INPUT(gradex);
    CHECK_INPUT(gradexs);

    return batch_norm_input_backward_cuda(input, gradoutput, gamma, ex, exs, gradex, gradexs, eps, cf);
}

std::vector<at::Tensor> batch_norm_collect_statistics(
    const at::Tensor input) {
    CHECK_INPUT(input);

    return batch_norm_collect_statistics_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_norm_transform_input", &batch_norm_transform_input, "BatchNorm forward (CUDA)");
  m.def("batch_norm_collect_grad_statistics", &batch_norm_collect_grad_statistics, "BatchNorm non-Input backward (CUDA)");
  m.def("batch_norm_input_backward", &batch_norm_input_backward, "BatchNorm Input backward (CUDA)");
  m.def("batch_norm_collect_statistics", &batch_norm_collect_statistics, "Expectation forward (CUDA)");
}
