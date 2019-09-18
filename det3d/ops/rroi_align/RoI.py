import math

import RotateRoIAlign_cuda as RRoI
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from torch.autograd import Function
from torch.nn.modules.utils import _pair


class RoIFunction(Function):
    @staticmethod
    def forward(
        ctx, inputs, rois, pooled_height, pooled_width, spatial_scale, sampling_ratio
    ):
        ctx.save_for_backward(rois)
        ctx.output_size = _pair((pooled_height, pooled_width))
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = inputs.size()
        output = RRoI.forward(
            inputs, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (rois,) = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        batch_size, channels, height, width = ctx.input_shape
        grad_input = RRoI.backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            batch_size,
            channels,
            height,
            width,
            sampling_ratio,
        )

        return grad_input, None, None, None, None, None


class RotateRoIAlign(nn.Module):
    def __init__(self, output_size, scale, ratio):
        super(RotateRoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = scale
        self.sampling_ratio = ratio

    def forward(self, inputs, rois):
        return RoIFunction.apply(
            inputs,
            rois,
            self.output_size[0],
            self.output_size[1],
            self.spatial_scale,
            self.sampling_ratio,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr
