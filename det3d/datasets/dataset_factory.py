from .kitti import KittiDataset
from .nuscenes import NuScenesDataset
from .lyft import LyftDataset
from .once import OnceDataset

dataset_factory = {
    "KITTI": KittiDataset,
    "NUSC": NuScenesDataset,
    "LYFT": LyftDataset,
    "ONCE": OnceDataset,
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
