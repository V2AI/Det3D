import torch
from lib.core.csrc.alignfeature.modules.align_feature import AlignFeature
from lib.core.csrc.correlation.modules.correlation import Correlation
from torch import nn


class Aggregation(nn.Module):
    def __init__(self, num_channel, name=""):
        super(Aggregation, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

    def forward(self, align_feature, feature):
        align_conv1 = self.conv1(align_feature)
        align_conv2 = self.conv2(align_conv1)
        align_conv3 = self.conv3(align_conv2)

        feature_conv1 = self.conv1(feature)
        feature_conv2 = self.conv2(feature_conv1)
        feature_conv3 = self.conv3(feature_conv2)

        weights = torch.cat([align_conv3, feature_conv3], dim=1)
        weights = torch.softmax(weights, dim=1)
        weights_slice = torch.split(weights, 1, dim=1)
        aggregation = weights_slice[0] * align_feature + weights_slice[1] * feature
        return aggregation


class Align_Feature_and_Aggregation(nn.Module):
    def __init__(self, num_channel, neighbor=9, name=""):
        super(Align_Feature_and_Aggregation, self).__init__()
        self.num_channel = num_channel
        self.embed_keyframe_conv = nn.Conv2d(num_channel, 64, 1)
        self.embed_current_conv = nn.Conv2d(num_channel, 64, 1)
        self.align_feature = AlignFeature(neighbor, neighbor)

        self.correlation = Correlation(
            kernel_size=1,
            patch_size=neighbor,
            stride=1,
            padding=0,
            dilation=1,
            dilation_patch=1,
        )
        self.aggregation = Aggregation(num_channel, name="Aggregation_Module")

    def forward(self, feature_select, feature_current):
        embed_feature_select = self.embed_keyframe_conv(feature_select)
        embed_feature_current = self.embed_current_conv(feature_current)

        weights = self.correlation(embed_feature_current, embed_feature_select)
        weights = weights.reshape(
            [weights.shape[0], -1, weights.shape[3], weights.shape[4]]
        )
        weights = torch.softmax(weights, dim=1)
        align_feature = self.align_feature(feature_select, weights)
        aggregation = self.aggregation(align_feature, feature_current)
        return aggregation
