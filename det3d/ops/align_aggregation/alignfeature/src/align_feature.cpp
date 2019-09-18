#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <iostream>

// declarations
int align_feature_cuda_forward_launcher(
    const at::Tensor data,
    const at::Tensor weight,
    const int weight_height,
    const int weight_width,
    const int N,
    const int C,
    const int Size_Weight,
    const int H,
    const int W,
    at::Tensor output);

int align_feature_cuda_backward_launcher(
    at::Tensor top_grad,
    at::Tensor data,
    at::Tensor weight,
    const int weight_height,
    const int weight_width,
    const int N,
    const int C,
    const int Size_Weight,
    const int H,
    const int W,
    at::Tensor grad_data,
    at::Tensor grad_weight);


#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int align_feature_cuda_forward(at::Tensor data, 
                               at::Tensor weight, 
                               int weight_height, 
                               int weight_width,
                               at::Tensor output) {


    CHECK_INPUT(data);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    int N = data.size(0);
    int C = data.size(1);
    int H = data.size(2);
    int W = data.size(3);
    int Size_Weight = weight.size(1);
    align_feature_cuda_forward_launcher(data, weight, weight_height, weight_width, N, C, Size_Weight, H, W, output);
    return 1;
}

int align_feature_cuda_backward(at::Tensor top_grad,
                                at::Tensor data,
                                at::Tensor weight,
                                int weight_height,
                                int weight_width,
                                int N,
                                int C,
                                int WeightSize,
                                int H,
                                int W,
                                at::Tensor grad_data,
                                at::Tensor grad_weight) {

    CHECK_INPUT(top_grad);
    CHECK_INPUT(data);
    CHECK_INPUT(weight);

    CHECK_INPUT(grad_data);
    CHECK_INPUT(grad_weight);

   
    align_feature_cuda_backward_launcher(top_grad, data, weight, weight_height, weight_width, 
                                         N, C, WeightSize, H, W, grad_data, grad_weight);

    return 1;
}



// C++ interfaces

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &align_feature_cuda_forward, "Align Feature Forward");
  m.def("backward", &align_feature_cuda_backward, "Align Feature Backward");
}
