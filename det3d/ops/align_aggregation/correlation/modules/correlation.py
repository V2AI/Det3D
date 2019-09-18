from torch.nn.modules.module import Module

from ..functions.correlation import CorrelationFunction


def correlation(
    input1, input2, kernel_size=1, patch_size=1, stride=1, padding=0, dilation_patch=1
):
    correlation_function = CorrelationFunction(
        kernel_size, patch_size, stride, padding, dilation_patch
    )
    return correlation_function(input1, input2)


class Correlation(Module):
    def __init__(
        self,
        kernel_size=1,
        patch_size=1,
        stride=1,
        padding=0,
        dilation=1,
        dilation_patch=1,
    ):
        super(Correlation, self).__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    def forward(self, input1, input2):
        return correlation(
            input1,
            input2,
            self.kernel_size,
            self.patch_size,
            self.stride,
            self.padding,
            self.dilation_patch,
        )
