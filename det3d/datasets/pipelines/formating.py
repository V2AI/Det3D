from det3d import torchie
import numpy as np
import torch

from ..registry import PIPELINES


class DataBundle(object):
    def __init__(self, data):
        self.data = data


@PIPELINES.register_module
class Reformat(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, res, info):
        meta = res["metadata"]
        points = res["lidar"]["points"]
        voxels = res["lidar"]["voxels"]
        anchors = res["lidar"]["targets"]["anchors"]

        data_bundle = dict(
            metadata=meta,
            points=points,
            voxels=voxels["voxels"],
            shape=voxels["shape"],
            num_points=voxels["num_points"],
            num_voxels=voxels["num_voxels"],
            coordinates=voxels["coordinates"],
            anchors=anchors,
        )

        if res["mode"] == "val":
            data_bundle.update(dict(metadata=meta,))

        calib = res.get("calib", None)
        if calib:
            data_bundle["calib"] = calib

        if res["mode"] != "test":
            annos = res["lidar"]["annotations"]
            data_bundle.update(annos=annos,)

        if res["mode"] == "train":
            ground_plane = res["lidar"].get("ground_plane", None)
            labels = res["lidar"]["targets"]["labels"]
            reg_targets = res["lidar"]["targets"]["reg_targets"]
            reg_weights = res["lidar"]["targets"]["reg_weights"]

            if ground_plane:
                data_bundle["ground_plane"] = ground_plane

            data_bundle.update(
                dict(labels=labels, reg_targets=reg_targets, reg_weights=reg_weights,)
            )

        return data_bundle, info


@PIPELINES.register_module
class PointCloudCollect(object):
    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "img_norm_cfg",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, info):

        results = info["res"]

        data = {}
        img_meta = {}

        for key in self.meta_keys:
            img_meta[key] = results[key]
        data["img_meta"] = DC(img_meta, cpu_only=True)

        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + "(keys={}, meta_keys={})".format(
            self.keys, self.meta_keys
        )
