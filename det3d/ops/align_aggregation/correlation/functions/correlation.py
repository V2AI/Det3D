import correlation_cuda
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair


def correlation(
    input1, input2, kernel_size=1, patch_size=1, stride=1, padding=0, dilation_patch=1
):
    correlation_function = CorrelationFunction(
        kernel_size, patch_size, stride, padding, dilation_patch
    )
    return correlation_function(input1, input2)


class CorrelationFunction(Function):
    def __init__(self, kernel_size, patch_size, stride, padding, dilation_patch):
        super(CorrelationFunction, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.patch_size = _pair(patch_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation_patch = _pair(dilation_patch)

    def forward(self, input1, input2):

        self.save_for_backward(input1, input2)
        kH, kW = self.kernel_size
        patchH, patchW = self.patch_size
        padH, padW = self.padding
        dilation_patchH, dilation_patchW = self.dilation_patch
        dH, dW = self.stride

        output = correlation_cuda.forward(
            input1,
            input2,
            kH,
            kW,
            patchH,
            patchW,
            padH,
            padW,
            dilation_patchH,
            dilation_patchW,
            dH,
            dW,
        )

        return output

    @once_differentiable
    def backward(self, grad_output):
        input1, input2 = self.saved_variables

        kH, kW = self.kernel_size
        patchH, patchW = self.patch_size
        padH, padW = self.padding
        dilation_patchH, dilation_patchW = self.dilation_patch
        dH, dW = self.stride

        grad_input1, grad_input2 = correlation_cuda.backward(
            input1,
            input2,
            grad_output,
            kH,
            kW,
            patchH,
            patchW,
            padH,
            padW,
            dilation_patchH,
            dilation_patchW,
            dH,
            dW,
        )
        return grad_input1, grad_input2


correlation = CorrelationFunction.apply
