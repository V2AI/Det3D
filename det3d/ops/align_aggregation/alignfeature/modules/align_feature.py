from torch.nn.modules.module import Module

from ..functions.align_feature import AlignFeatureFunction


def align_feature(data, weight, weight_height, weight_width):
    align_feature_function = AlignFeatureFunction(weight_height, weight_width)
    return align_feature_function(data, weight)


class AlignFeature(Module):
    def __init__(self, weight_height, weight_width):
        super(AlignFeature, self).__init__()
        self.weight_height = weight_height
        self.weight_width = weight_width

    def forward(self, data, weight):
        return align_feature(data, weight, self.weight_height, self.weight_width)
