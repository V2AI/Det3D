#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be continguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor ROIAlign_forward_cuda(const at::Tensor& input, const at::Tensor& rois,
		const float spatial_scale, const int pooled_height, const int pooled_width,
		const int sampling_ratio);



at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad, const at::Tensor& rois,
		const float spatial_scale, const int pooled_height, const int pooled_width,
		const int batch_size, const int channels, const int height, const int width,
		const int sampling_ratio);

at::Tensor ROIAlign_forward(const at::Tensor& input, const at::Tensor& rois,
		const float spatial_scale, const int pooled_height, const int pooled_width,
		const int sampling_ratio) {
	CHECK_INPUT(input);
	CHECK_INPUT(rois);
	return ROIAlign_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

at::Tensor ROIAlign_backward(const at::Tensor& grad, const at::Tensor& rois,
		const float spatial_scale, const int pooled_height, const int pooled_width,
		const int batch_size, const int channels, const int height, const int width,
		const int sampling_ratio) {
	CHECK_INPUT(grad);
	CHECK_INPUT(rois);
	return ROIAlign_backward_cuda(grad, rois, spatial_scale, pooled_height, pooled_width, 
				batch_size, channels, height, width, sampling_ratio);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ROIAlign_forward, "RotateRoIAlign forward (CUDA)");
  m.def("backward", &ROIAlign_backward, "RotateRoIAlign backward (CUDA)");
}


