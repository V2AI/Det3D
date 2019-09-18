import numpy as np
from pathlib import Path
from pyquaternion import Quaternion
from tqdm import tqdm
import pickle

from lyft_dataset_sdk.lyftdataset import LyftDataset

import det3d.datasets.lyft.splits as splits


def quaternion_yaw(q: Quaternion) -> float:
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    yaw = np.arctan2(v[1], v[0])
    return yaw


def _get_available_scenes(lyft):
    available_scenes = []
    print("total scene num:", len(lyft.scene))
    for scene in lyft.scene:
        scene_token = scene["token"]
        scene_rec = lyft.get("scene", scene_token)
        sample_rec = lyft.get("sample", scene_rec["first_sample_token"])
        sd_rec = lyft.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = lyft.get_sample_data(sd_rec["token"])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            if not sd_rec["next"] == "":
                sd_rec = lyft.get("sample_data", sd_rec["next"])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def get_sample_data(lyft, sample_data_token):
    sd_rec = lyft.get("sample_data", sample_data_token)
    cs_rec = lyft.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])

    sensor_rec = lyft.get("sensor", cs_rec["sensor_token"])
    pose_rec = lyft.get("ego_pose", sd_rec["ego_pose_token"])

    boxes = lyft.get_boxes(sample_data_token)

    box_list = []
    for box in boxes:
        box.translate(-np.array(pose_rec["translation"]))
        box.rotate(Quaternion(pose_rec["rotation"]).inverse)

        box.translate(-np.array(cs_rec["translation"]))
        box.rotate(Quaternion(cs_rec["rotation"]).inverse)

        box_list.append(box)

    return box_list, pose_rec


def _fill_trainval_infos(lyft, train_scenes, val_scenes, test=False):
    from lyft_dataset_sdk.utils.geometry_utils import transform_matrix

    train_lyft_infos = []
    val_lyft_infos = []

    ref_chans = ["LIDAR_TOP", "LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]

    for sample in tqdm(lyft.sample):
        ref_info = {}
        for ref_chan in ref_chans:
            if ref_chan not in sample["data"]:
                continue
            ref_sd_token = sample["data"][ref_chan]
            ref_sd_rec = lyft.get("sample_data", ref_sd_token)
            ref_cs_token = ref_sd_rec["calibrated_sensor_token"]
            ref_cs_rec = lyft.get("calibrated_sensor", ref_cs_token)

            ref_to_car = transform_matrix(
                ref_cs_rec["translation"],
                Quaternion(ref_cs_rec["rotation"]),
                inverse=False,
            )

            ref_from_car = transform_matrix(
                ref_cs_rec["translation"],
                Quaternion(ref_cs_rec["rotation"]),
                inverse=True,
            )

            ref_lidar_path = lyft.get_sample_data_path(ref_sd_token)

            ref_info[ref_chan] = {
                "lidar_path": ref_lidar_path,
                "ref_from_car": ref_from_car,
                "ref_to_car": ref_to_car,
            }

            if ref_chan == "LIDAR_TOP":
                ref_boxes, ref_pose_rec = get_sample_data(lyft, ref_sd_token)
                ref_time = 1e-6 * ref_sd_rec["timestamp"]
                car_from_global = transform_matrix(
                    ref_pose_rec["translation"],
                    Quaternion(ref_pose_rec["rotation"]),
                    inverse=True,
                )

        info = {
            "ref_info": ref_info,
            "token": sample["token"],
            "car_from_global": car_from_global,
            "timestamp": ref_time,
        }

        if not test:
            annotations = [
                lyft.get("sample_annotation", token) for token in sample["anns"]
            ]

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(
                -1, 1
            )
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes]).reshape(-1, 1)
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

            assert len(annotations) == len(gt_boxes)

            info["gt_boxes"] = gt_boxes
            info["gt_boxes_velocity"] = velocity
            info["gt_names"] = names
            info["gt_boxes_token"] = tokens

        if sample["scene_token"] in train_scenes:
            train_lyft_infos.append(info)
        else:
            val_lyft_infos.append(info)

    return train_lyft_infos, val_lyft_infos


def create_lyft_infos(root_path, version="trainval"):
    available_vers = ["trainval", "test"]
    assert version in available_vers
    if version == "trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "test":
        train_scenes = splits.test
        val_scenes = []
    else:
        raise ValueError("unknown")
    test = "test" in version
    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    root_path = Path(root_path)
    data_path = root_path / f"{version}"
    json_path = root_path / f"{version}/data"
    lyft = LyftDataset(data_path, json_path)

    available_scenes = _get_available_scenes(lyft)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )

    train_lyft_infos, val_lyft_infos = _fill_trainval_infos(
        lyft, train_scenes, val_scenes, test
    )

    if test:
        print(f"test sample: {len(train_lyft_infos)}")
        with open(data_path / "lyft_info_test.pkl", "wb") as f:
            pickle.dump(train_lyft_infos, f)
    else:
        print(f"train sample: {len(train_lyft_infos)}")
        print(f"val sample: {len(val_lyft_infos)}")
        with open(data_path / "lyft_info_train.pkl", "wb") as f:
            pickle.dump(train_lyft_infos, f)
        with open(data_path / "lyft_info_val.pkl", "wb") as f:
            pickle.dump(val_lyft_infos, f)
