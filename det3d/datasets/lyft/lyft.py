import numpy as np
import pickle

from copy import deepcopy

from det3d.core import box_np_ops
from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS
from .eval import get_lyft_eval_result


@DATASETS.register_module
class LyftDataset(PointCloudDataset):

    NumPointFeatures = 5
    DatasetName = "LyftDataset"

    def __init__(
        self,
        root_path,
        info_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        **kwargs
    ):
        super(LyftDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode
        )

        self._info_path = info_path
        self._class_names = class_names

        self.load_infos(self._info_path)
        self._num_point_features = __class__.NumPointFeatures

        self._cls2label = {}
        self._label2cls = {}
        for i in range(len(self._class_names)):
            self._cls2label[self._class_names[i]] = i
            self._label2cls[i] = self._class_names[i]

    def load_infos(self, info_path):

        with open(self._info_path, "rb") as f:
            _lyft_infos_all = pickle.load(f)

        if not self.test_mode:  # if training
            self.frac = int(len(_lyft_infos_all) * 0.25)

            _cls_infos = {name: [] for name in self._class_names}
            for info in _lyft_infos_all:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
            _cls_dist = {k: len(v) / duplicated_samples for k, v in _cls_infos.items()}

            self._lyft_infos = []

            frac = 1.0 / len(self._class_names)
            ratios = [frac / v for v in _cls_dist.values()]

            for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
                self._lyft_infos += np.random.choice(
                    cls_infos, int(len(cls_infos) * ratio)
                ).tolist()

            _cls_infos = {name: [] for name in self._class_names}
            for info in self._lyft_infos:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            _cls_dist = {
                k: len(v) / len(self._lyft_infos) for k, v in _cls_infos.items()
            }
        else:
            if isinstance(_lyft_infos_all, dict):
                self._lyft_infos = []
                for v in _lyft_infos_all.values():
                    self._lyft_infos.extend(v)
            else:
                self._lyft_infos = _lyft_infos_all

    def __len__(self):
        with open(self._info_path, "rb") as f:
            self._lyft_infos = pickle.load(f)
        return len(self._lyft_infos)

    @property
    def num_point_features(self):
        return self._num_point_features

    @property
    def ground_truth_annotations(self):
        annos = []
        for i in range(len(self._lyft_infos)):
            info = self._lyft_infos[i]
            token = info["token"]
            anno = {}
            gt_mask = np.where(
                np.array(
                    [1 if cls in self._cls2label else 0 for cls in info["gt_names"]]
                )
                == 1
            )[0]
            gt_boxes = info["gt_boxes"][gt_mask]
            box_num = gt_boxes.shape[0]
            anno["bbox"] = np.zeros((box_num, 4))
            anno["alpha"] = np.zeros(box_num, dtype=np.float32)
            anno["location"] = gt_boxes[:, :3]
            anno["dimensions"] = gt_boxes[:, 3:6]
            anno["rotation_y"] = gt_boxes[:, -1]
            anno["name"] = info["gt_names"][gt_mask].tolist()
            anno["gt_labels"] = np.array([self._cls2label[cls] for cls in anno["name"]])
            annos.append(anno)
        return annos

    def get_sensor_data(self, idx):

        info = self._lyft_infos[idx]

        res = {
            "lidar": {"type": "lidar", "points": None,},
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def convert_detection_to_lyft_annos(self, dt_annos):
        annos = []
        for token, dt_anno in dt_annos.items():
            anno = {}
            dt_boxes = dt_anno["box3d_lidar"].cpu().numpy()
            box_num = dt_boxes.shape[0]
            labels = dt_anno["label_preds"].cpu().numpy()
            scores = dt_anno["scores"].cpu().numpy()
            anno["score"] = scores
            anno["bbox"] = np.zeros((box_num, 4))
            anno["alpha"] = np.zeros(box_num, dtype=np.float32)
            anno["dimensions"] = dt_boxes[:, 3:6]
            anno["location"] = dt_boxes[:, :3]
            anno["rotation_y"] = dt_boxes[:, -1]
            anno["name"] = [self._label2cls[label] for label in labels]
            annos.append(anno)

        return annos

    def evaluation(self, detections, output_dir=None):
        gt_annos = self.ground_truth_annotations
        dt_annos = self.convert_detection_to_lyft_annos(detections)
        result_lyft = get_lyft_eval_result(gt_annos, dt_annos, self._class_names)
        return result_lyft_dict
