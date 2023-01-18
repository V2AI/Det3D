import sys
import pickle
import json
import random
import operator
from numba.cuda.simulator.api import detect
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

from det3d.datasets.custom import PointCloudDataset

from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class OnceDataset(PointCloudDataset):
    NumPointFeatures = 4 # x, y, z, reflection

    def __init__(
        self,
        info_path,
        root_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        sample=False,
        load_interval=1,
        **kwargs,
    ):
        self.load_interval = load_interval 
        self.sample = sample
        super(OnceDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self._info_path = info_path
        self._class_names = class_names
        self._num_point_features = OnceDataset.NumPointFeatures

    def __len__(self):
        if not hasattr(self, "_once_infos"):
            with open(self._info_path, "rb") as f:
                self._once_infos = pickle.load(f)
        return len(self._once_infos)

    def __getitem__(self, idx):
        return self.get_sensor_data(idx, with_gp=True)

    def get_sensor_data(self, idx, with_image=False):
        info = self._once_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "num_point_features": OnceDataset.NumPointFeatures,
                "image_idx": info['frame_id'],
                "image_shape": info['metadata']['image_size'],
            },
        "mode": "val" if self.test_mode else "train",
        },

        data, _ = self.pipeline(res, info)
        
        return data

    @property
    def num_point_features(self, idx):
        return self._num_point_features

    @property
    def ground_truth_annotations(self):
        if "annos" not in self._once_infos[0]:
            return None
        
        gt_annos = [info["annos"] for info in self._once_infos]

        return gt_annos
    
