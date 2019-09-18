import copy
from pathlib import Path
import pickle

import fire

from det3d.datasets.kitti import kitti_common as kitti_ds
from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.lyft import lyft_common as lyft_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database


def kitti_data_prep(root_path):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database(
        "KITTI", root_path, Path(root_path) / "kitti_infos_train.pkl"
    )


def nuscenes_data_prep(root_path, version, nsweeps=10):
    nu_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps)
    create_groundtruth_database(
        "NUSC",
        root_path,
        Path(root_path) / "infos_train_{:02d}sweeps_withvelo.pkl".format(nsweeps),
        nsweeps=nsweeps,
    )
    # nu_ds.get_sample_ground_plane(root_path, version=version)


def lyft_data_prep(root_path, version="trainval"):
    lyft_ds.create_lyft_infos(root_path, version=version)
    create_groundtruth_database(
        "LYFT", root_path, Path(root_path) / "lyft_info_train.pkl"
    )


if __name__ == "__main__":
    fire.Fire()
