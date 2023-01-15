import pickle
import json
import numpy as np

from pathlib import Path
from tqdm import tqdm

from det3d.core import box_np_ops


def create_once_infos(data_path, save_path=None, relative_path=True):

    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    splits = ['train', 'val', 'test']
    for split in splits:
        once_infos = get_infos(data_path, split)
        filename = save_path / f'once_infos_{split}.pkl'
        print(f"Once info {split} file is saved to {filename}")
        with open(filename, "wb") as f:
            pickle.dump(once_infos, f)


def get_infos(data_path, split):
    if isinstance(data_path, str):
        data_path = Path(data_path)
    split_dir = data_path / 'ImageSets' / (split + '.txt')
    sample_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    infos = []
    for seq_idx in tqdm(sample_list):
        print('%s seq_idx: %s' % (split, seq_idx))
        seq_info = []
        seq_path = Path(data_path) / 'data' / seq_idx
        json_path = seq_path / ('%s.json' % seq_idx)
        with open(json_path, 'r') as f:
            info_seq = json.load(f)
        meta_info = info_seq['meta_info']
        calib = info_seq['calib']
        for frame_idx, frame in enumerate(info_seq['frames']):
            frame_id = frame['frame_id']
            if frame_idx == 0:
                prev_id = None
            else:
                prev_id = info_seq['frames'][frame_idx-1]['frame_id']
            if frame_idx == len(info_seq['frames'])-1:
                next_id = None
            else:
                next_id = info_seq['frames'][frame_idx+1]['frame_id']
            pc_path = str(seq_path / 'lidar_roof' / ('%s.bin' % frame_id))
            pose = np.array(frame['pose'])
            frame_dict = {
                'sequence_id': seq_idx,
                'frame_id': frame_id,
                'timestamp': int(frame_id),
                'prev_id': prev_id,
                'next_id': next_id,
                'meta_info': meta_info,
                'lidar': pc_path,
                'pose': pose
            }
            calib_dict = {}
            cam_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
            for cam_name in cam_names:
                cam_path = str(seq_path / cam_name / ('%s.jpg' % frame_id))
                frame_dict.update({cam_name: cam_path})
                calib_dict[cam_name] = {}
                calib_dict[cam_name]['cam_to_velo'] = np.array(calib[cam_name]['cam_to_velo'])
                calib_dict[cam_name]['cam_intrinsic'] = np.array(calib[cam_name]['cam_intrinsic'])
                calib_dict[cam_name]['distortion'] = np.array(calib[cam_name]['distortion'])
            frame_dict.update({'calib': calib_dict})

            if 'annos' in frame:
                annos = frame['annos']
                # boxes_3d of shape `(N, 7)` where
                # N is the number of objects
                # 7 is `(x, y, z, h, w, l, theta)` in lidar_coordinates
                boxes_3d = np.array(annos['boxes_3d'])
                if boxes_3d.shape[0] == 0:
                    print(f"Skipping {frame_id} of {seq_idx} since no objects annotated.")
                    continue
                boxes_2d_dict = {}
                for cam_name in cam_names:
                    boxes_2d_dict[cam_name] = np.array(annos['boxes_2d'][cam_name])
                annos_dict = {
                    'name': np.array(annos['names']),
                    'boxes_3d': boxes_3d,
                    'boxes_2d': boxes_2d_dict
                }

                # caculate num points in gt
                bin_path = data_path / seq_idx / 'lidar_roof' / '{}.bin'.format(frame_id)
                points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
                indices = box_np_ops.points_in_rbbox(points[:, :3], annos_dict['boxes_3d'])
                num_points_in_gt = indices.sum(0)
                annos_dict['num_points_in_gt'] = num_points_in_gt.astype(np.int32)

                frame_dict.update({'annos': annos_dict})

            seq_info.append(frame_dict)
        return seq_info
        
    return infos
