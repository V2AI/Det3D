# Det3D

A general 3D Object Detection codebase in PyTorch

## Call for contribution.
* Support Waymo Dataset.
* Add other 3D detection / segmentation models, such as VoteNet, STD, etc.

## Introduction

Det3D is the first 3D Object Detection toolbox which provides off the box implementations of many 3D object detection algorithms such as PointPillars, SECOND, PIXOR, etc, as well as state-of-the-art methods on major benchmarks like KITTI(ViP) and nuScenes(CBGS). Key features of Det3D include the following aspects:

* Multi Datasets Support: KITTI, nuScenes, Lyft
* Point-based and Voxel-based model zoo
* State-of-the-art performance
* DDP & SyncBN


## Installation

Please refer to [INSTALATION.md](INSTALLATION.md).

## Quick Start

Please refer to [GETTING_STARTED.md](GETTING_STARTED.md).

## Model Zoo and Baselines

### [Second](examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py) on KITTI(val) Dataset

```
car  AP @0.70, 0.70,  0.70:
bbox AP:90.54, 89.35, 88.43
bev  AP:89.89, 87.75, 86.81
3d   AP:87.96, 78.28, 76.99
aos  AP:90.34, 88.81, 87.66
```

### [PointPillars](examples/point_pillars/configs/kitti_point_pillars_mghead_syncbn.py) on KITTI(val) Dataset

```	
car  AP@0.70,  0.70,  0.70:
bbox AP:90.63, 88.86, 87.35
bev  AP:89.75, 86.15, 83.00
3d   AP:85.75, 75.68, 68.93
aos  AP:90.48, 88.36, 86.58
```

### To Be Released

1. [PointPillars](examples/point_pillars/configs/nusc_all_point_pillars_mghead_syncbn.py) on NuScenes(val) Dataset
2. [CGBS](examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py) on NuScenes(val) Dataset
3. [CGBS](examples/cbgs/configs/lyft_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py) on Lyft(val) Dataset

## Currently Support

* Models
  - [x] VoxelNet
  - [x] SECOND
  - [x] PointPillars
* Features
    - [x] Multi task learning & Multi-task Learning
    - [x] Distributed Training and Validation
    - [x] SyncBN
    - [x] Flexible anchor dimensions
    - [x] TensorboardX
    - [x] Checkpointer & Breakpoint continue
    - [x] Self-contained visualization
    - [x] Finetune
    - [x] Multiscale Training & Validation
    - [x] Rotated RoI Align


## TODO List
* Models
  - [ ] PointRCNN
  - [ ] PIXOR

## Developers

[Benjin Zhu](https://github.com/poodarchu/) , [Bingqi Ma](https://github.com/a157801)

## License

Det3D is released under the [Apache licenes](LICENES).

## Acknowledgement

* [mmdetection](https://github.com/open-mmlab/mmdetection) 
* [mmcv](https://github.com/open-mmlab/mmcv)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
