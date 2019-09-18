from torch.autograd import Function
import torch
import align_feature_cuda


class AlignFeatureFunction(Function):
    def __init__(self, weight_width, weight_height):
        super(AlignFeatureFunction, self).__init__()
        self.weight_width = weight_width
        self.weight_height = weight_height

    def forward(self, data, weight):
        self.save_for_backward(data, weight)
        N = data.size(0)
        C = data.size(1)
        H = data.size(2)
        W = data.size(3)
        output = data.new_zeros(N, C, H, W)
        if data.is_cuda:
            align_feature_cuda.forward(
                data, weight, self.weight_height, self.weight_width, output
            )
        else:
            raise NotImplementedError
        return output

    def backward(self, grad_output):
        data, weight = self.saved_variables
        N, C, H, W = data.size()
        Weight_Size = weight.size(1)
        grad_data = data.new_zeros(N, C, H, W)
        grad_weight = weight.new_zeros(N, Weight_Size, H, W)

        if data.is_cuda:
            align_feature_cuda.backward(
                grad_output,
                data,
                weight,
                self.weight_height,
                self.weight_width,
                N,
                C,
                Weight_Size,
                H,
                W,
                grad_data,
                grad_weight,
            )
        else:
            raise NotImplementedError

        return grad_data, grad_weight
