import numpy as np
import pickle
import os

from copy import deepcopy

from det3d.core import box_np_ops
from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS

from .kitti_common import *
from .eval import get_official_eval_result, get_coco_eval_result


@DATASETS.register_module
class KittiDataset(PointCloudDataset):

    NumPointFeatures = 4

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
        super(KittiDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode
        )
        assert self._info_path is not None
        if not hasattr(self, "_kitti_infos"):
            with open(self._info_path, "rb") as f:
                infos = pickle.load(f)
            self._kitti_infos = infos
        self._num_point_features = __class__.NumPointFeatures
        # print("remain number of infos:", len(self._kitti_infos))
        self._class_names = class_names
        self.plane_dir = "/data/Datasets/KITTI/Kitti/object/training/planes"

    def __len__(self):
        if not hasattr(self, "_kitti_infos"):
            with open(self._info_path, "rb") as f:
                self._kitti_infos = pickle.load(f)

        return len(self._kitti_infos)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, "%06d.txt" % idx)
        with open(plane_file, "r") as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @property
    def num_point_features(self):
        return self._num_point_features

    @property
    def ground_truth_annotations(self):
        if "annos" not in self._kitti_infos[0]:
            return None

        gt_annos = [info["annos"] for info in self._kitti_infos]

        return gt_annos

    def convert_detection_to_kitti_annos(self, detection):
        class_names = self._class_names
        det_image_idxes = [k for k in detection.keys()]
        gt_image_idxes = [str(info["image"]["image_idx"]) for info in self._kitti_infos]
        # print(f"det_image_idxes: {det_image_idxes[:10]}")
        # print(f"gt_image_idxes: {gt_image_idxes[:10]}")
        annos = []
        # for i in range(len(detection)):
        for det_idx in gt_image_idxes:
            det = detection[det_idx]
            info = self._kitti_infos[gt_image_idxes.index(det_idx)]
            # info = self._kitti_infos[i]
            calib = info["calib"]
            rect = calib["R0_rect"]
            Trv2c = calib["Tr_velo_to_cam"]
            P2 = calib["P2"]
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()

            anno = get_start_result_anno()
            num_example = 0

            if final_box_preds.shape[0] != 0:
                final_box_preds[:, -1] = box_np_ops.limit_period(
                    final_box_preds[:, -1], offset=0.5, period=np.pi * 2,
                )
                final_box_preds[:, 2] -= final_box_preds[:, 5] / 2

                # aim: x, y, z, w, l, h, r -> -y, -z, x, h, w, l, r
                # (x, y, z, w, l, h r) in lidar -> (x', y', z', l, h, w, r) in camera
                box3d_camera = box_np_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c
                )
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_np_ops.center_to_corner_box3d(
                    box3d_camera[:, :3],
                    box3d_camera[:, 3:6],
                    box3d_camera[:, 6],
                    camera_box_origin,
                    axis=1,
                )
                box_corners_in_image = box_np_ops.project_to_image(box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = np.min(box_corners_in_image, axis=1)
                maxxy = np.max(box_corners_in_image, axis=1)
                bbox = np.concatenate([minxy, maxxy], axis=1)

                for j in range(box3d_camera.shape[0]):
                    image_shape = info["image"]["image_shape"]
                    if bbox[j, 0] > image_shape[1] or bbox[j, 1] > image_shape[0]:
                        continue
                    if bbox[j, 2] < 0 or bbox[j, 3] < 0:
                        continue
                    bbox[j, 2:] = np.minimum(bbox[j, 2:], image_shape[::-1])
                    bbox[j, :2] = np.maximum(bbox[j, :2], [0, 0])
                    anno["bbox"].append(bbox[j])

                    anno["alpha"].append(
                        -np.arctan2(-final_box_preds[j, 1], final_box_preds[j, 0])
                        + box3d_camera[j, 6]
                    )
                    # anno["dimensions"].append(box3d_camera[j, [4, 5, 3]])
                    anno["dimensions"].append(box3d_camera[j, 3:6])
                    anno["location"].append(box3d_camera[j, :3])
                    anno["rotation_y"].append(box3d_camera[j, 6])
                    anno["name"].append(class_names[int(label_preds[j])])
                    anno["truncated"].append(0.0)
                    anno["occluded"].append(0)
                    anno["score"].append(scores[j])

                    num_example += 1

            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = det["metadata"]
        return annos

    def evaluation(self, detections, output_dir=None):
        """
        detection
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        """
        gt_annos = self.ground_truth_annotations
        dt_annos = self.convert_detection_to_kitti_annos(detections)

        # firstly convert standard detection to kitti-format dt annos
        z_axis = 1  # KITTI camera format use y as regular "z" axis.
        z_center = 1.0  # KITTI camera box's center is [0.5, 1, 0.5]
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.

        result_official_dict = get_official_eval_result(
            gt_annos, dt_annos, self._class_names, z_axis=z_axis, z_center=z_center
        )
        result_coco_dict = get_coco_eval_result(
            gt_annos, dt_annos, self._class_names, z_axis=z_axis, z_center=z_center
        )

        results = {
            "results": {
                "official": result_official_dict["result"],
                "coco": result_coco_dict["result"],
            },
            "detail": {
                "eval.kitti": {
                    "official": result_official_dict["detail"],
                    "coco": result_coco_dict["detail"],
                }
            },
        }

        return results, dt_annos

    def __getitem__(self, idx):
        return self.prepare_input(idx, with_gp=True)

    def prepare_input(self, idx, with_image=False, with_gp=False):

        info = self._kitti_infos[idx]

        if with_gp:
            gp = self.get_road_plane(idx)

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "ground_plane": -gp[-1] if with_gp else None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": KittiDataset.NumPointFeatures,
                "image_idx": info["image"]["image_idx"],
                "image_shape": info["image"]["image_shape"],
                "token": str(info["image"]["image_idx"]),
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
        }

        data, _ = self.pipeline(res, info)

        # objgraph.show_growth(limit=3)
        # objgraph.get_leaking_objects()

        image_info = info["image"]
        image_path = image_info["image_path"]

        if with_image:
            image_path = self._root_path / image_path
            with open(str(image_path), "rb") as f:
                image_str = f.read()
            data["cam"] = {
                "type": "camera",
                "data": image_str,
                "datatype": "png",
            }

        return data
