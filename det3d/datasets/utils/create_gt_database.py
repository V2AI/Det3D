import pickle
from pathlib import Path

import numpy as np

from det3d.core import box_np_ops
from det3d.datasets.dataset_factory import get_dataset

from tqdm import tqdm

dataset_name_map = {
    "KITTI": "KittiDataset",
    "NUSC": "NuScenesDataset",
    "LYFT": "LyftDataset",
    "ONCE": "OnceDataset",
}


def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_path=None,
    used_classes=None,
    db_path=None,
    dbinfo_path=None,
    relative_path=True,
    add_rgb=False,
    lidar_only=False,
    bev_only=False,
    coors_range=None,
    **kwargs,
):
    pipeline = [
        {
            "type": "LoadPointCloudFromFile",
            "dataset": dataset_name_map[dataset_class_name],
        },
        {"type": "LoadPointCloudAnnotations", "with_bbox": True},
    ]

    if "nsweeps" in kwargs:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path,
            root_path=data_path,
            pipeline=pipeline,
            test_mode=True,
            nsweeps=kwargs["nsweeps"],
        )
        nsweeps = dataset.nsweeps
    else:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path, root_path=data_path, test_mode=True, pipeline=pipeline
        )
        nsweeps = 1

    root_path = Path(data_path)

    if dataset_class_name == "NUSC":
        if db_path is None:
            db_path = root_path / f"gt_database_{nsweeps}sweeps_withvelo"
        if dbinfo_path is None:
            dbinfo_path = root_path / f"dbinfos_train_{nsweeps}sweeps_withvelo.pkl"
    else:
        if db_path is None:
            db_path = root_path / "gt_database"
        if dbinfo_path is None:
            dbinfo_path = root_path / "dbinfos_train.pkl"
    if dataset_class_name == "NUSC" or dataset_class_name == "LYFT":
        point_features = 5
    elif dataset_class_name == "KITTI" or dataset_class_name == "ONCE":
        point_features = 4

    db_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    group_counter = 0

    # def prepare_single_data(index):
    for index in tqdm(range(len(dataset))):
        image_idx = index
        sensor_data = dataset.get_sensor_data(index)
        if sensor_data is None:
            continue
        
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]

        if dataset_class_name == "NUSC":
            points = sensor_data["lidar"]["combined"]
        elif dataset_class_name == "KITTI":
            points = sensor_data["lidar"]["points"]
        elif dataset_class_name == "LYFT":
            points = sensor_data["lidar"]["points"]
        elif dataset_class_name == "ONCE":
            points = sensor_data["lidar"]["points"]

        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]
        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            filepath = db_path / filename
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, "w") as f:
                gt_points[:, :point_features].tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_dump_path = str(db_path.stem + "/" + filename)
                else:
                    db_dump_path = str(filepath)

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
        # print(f"Finish {index}th sample")

    print("dataset length: ", len(dataset))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, "wb") as f:
        pickle.dump(all_db_infos, f)
