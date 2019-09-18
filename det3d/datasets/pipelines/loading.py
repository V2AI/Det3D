import os.path as osp
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
from det3d.datasets.kitti import kitti_common as kitti

from ..registry import PIPELINES


def read_file(path, tries=2, num_point_feature=4):
    points = None
    try_cnt = 0
    while points is None and try_cnt < tries:
        try_cnt += 1
        try:
            points = np.fromfile(path, dtype=np.float32)
            s = points.shape[0]
            if s % 5 != 0:
                points = points[: s - (s % 5)]
            points = points.reshape(-1, 5)[:, :num_point_feature]
        except Exception:
            points = None

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep):
    min_distance = 1.0
    # points_sweep = np.fromfile(str(sweep["lidar_path"]),
    #                            dtype=np.float32).reshape([-1,
    #                                                       5])[:, :4].T
    points_sweep = read_file(str(sweep["lidar_path"])).T

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    # points_sweep[3, :] /= 255
    points_sweep = remove_close(points_sweep, min_distance)
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "KittiDataset":

            pc_info = info["point_cloud"]
            velo_path = Path(pc_info["velodyne_path"])
            if not velo_path.is_absolute():
                velo_path = (
                    Path(res["metadata"]["image_prefix"]) / pc_info["velodyne_path"]
                )
            velo_reduced_path = (
                velo_path.parent.parent
                / (velo_path.parent.stem + "_reduced")
                / velo_path.name
            )
            if velo_reduced_path.exists():
                velo_path = velo_reduced_path
            points = np.fromfile(str(velo_path), dtype=np.float32, count=-1).reshape(
                [-1, res["metadata"]["num_point_features"]]
            )

            res["lidar"]["points"] = points

        elif self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path))

            # points[:, 3] /= 255
            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) <= len(
                info["sweeps"]
            ), "nsweeps {} should not greater than list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep)
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])

        elif self.type == "LyftDataset":

            top_info = info["ref_info"]["LIDAR_TOP"]
            points = read_file(top_info["lidar_path"])

            # if "LIDAR_FRONT_LEFT" in info["ref_info"] and \
            #         "LIDAR_FRONT_RIGHT" in info["ref_info"]:
            #
            #     def lr2top(lr_info):
            #         lr_point = read_file(lr_info["lidar_path"],
            #                              num_point_features=3)
            #         lr_point_xyz = lr_point[:, :3]
            #         lr_point_f = lr_point[:, 3:]
            #         trans = reduce(
            #             np.dot,
            #             [top_info['ref_from_car'], lr_info['ref_to_car']])
            #         lrpoints = trans.dot(
            #             np.vstack(
            #                 (lr_point_xyz.T, np.ones(lr_point.shape[0])))).T
            #         lrpoints = np.hstack((lrpoints[:, :3], lr_point_f))
            #         return lrpoints
            #
            #     left_info = info['ref_info']['LIDAR_FRONT_LEFT']
            #     right_info = info['ref_info']['LIDAR_FRONT_RIGHT']
            #
            #     left_point = lr2top(left_info)
            #     right_point = lr2top(right_info)
            #     points = np.concatenate((points, left_point, right_point),
            #                             axis=0).astype(np.float32)

            res["lidar"]["points"] = points

        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset", "LyftDataset"] and "gt_boxes" in info:

            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }

        elif res["type"] == "KittiDataset":

            calib = info["calib"]
            calib_dict = {
                "rect": calib["R0_rect"],
                "Trv2c": calib["Tr_velo_to_cam"],
                "P2": calib["P2"],
            }
            res["calib"] = calib_dict
            if "annos" in info:
                annos = info["annos"]
                # we need other objects to avoid collision when sample
                annos = kitti.remove_dontcare(annos)
                locs = annos["location"]
                dims = annos["dimensions"]
                rots = annos["rotation_y"]
                gt_names = annos["name"]
                gt_boxes = np.concatenate(
                    [locs, dims, rots[..., np.newaxis]], axis=1
                ).astype(np.float32)
                calib = info["calib"]
                gt_boxes = box_np_ops.box_camera_to_lidar(
                    gt_boxes, calib["R0_rect"], calib["Tr_velo_to_cam"]
                )

                # only center format is allowed. so we need to convert
                # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
                box_np_ops.change_box3d_center_(
                    gt_boxes, [0.5, 0.5, 0], [0.5, 0.5, 0.5]
                )

                res["lidar"]["annotations"] = {
                    "boxes": gt_boxes,
                    "names": gt_names,
                }
                res["cam"]["annotations"] = {
                    "boxes": annos["bbox"],
                    "names": gt_names,
                }
        else:
            return NotImplementedError

        return res, info
