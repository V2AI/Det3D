# Det3D

A general 3D Object Detection codebase in PyTorch

## Introduction

Det3D is the first 3D Object Detection toolbox which provides off the box implementations of many 3D object detection algorithms such as PointPillars, SECOND, PointRCNN, PIXOR, etc, as well as state-of-the-art methods on major benchmarks like KITTI(ViP) and nuScenes(CBGS). Key features of Det3D include the following apects:

* Multi Datasets Support: KITTI, nuScenes, Lyft, waymo
* Point-based and Voxel-based model zoo
* State-of-the-art performance
* DDP & SyncBN

Project Organization
------------
```
.
├── examples   -- all supported models
├── docs
├── det3d
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
└── tools
```

Prerequisite
-------------------
- Cuda 9.0 +
- Pytorch 1.1
- Python 3.6+
- [APEX](https://github.com/NVIDIA/apex.git)
- [spconv](https://github.com/traveller59/spconv/commit/73427720a539caf9a44ec58abe3af7aa9ddb8e39) 
- nuscenes_devkit
- Lyft_dataset_kit

## Get Started
```
git clone https://github.com/poodarchu/det3d.git
python setup.py build develop
```
## Data Preparation

-----------------
###  1. download data and organise as follows
```
# For KITTI Dataset
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory

# For nuScenes Dataset         
└── NUSCENES_TRAINVAL_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       └── v1.0-trainval <-- metadata and annotations
└── NUSCENES_TEST_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       └── v1.0-test     <-- metadata
```
### 2. Convert to pkls
```
# KITTI
python create_data.py kitti_data_prep --root_path=KITTI_DATASET_ROOT
# nuScenes
python create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10'
# Lyft
python create_data.py lyft_data_prep --root_path=LYFT_TEST_DATASET_ROOT
```
### 3. Modify configs
Modify dataset pkl file path in src/configs/xxx.config:
```
DATASET:
    TYPE: nuScenes
    ROOT_PATH: /data/Datasets/nuScenes
    INFO_PATH: /data/Datasets/nuScenes/infos_train_10sweeps_withvelo.pkl
    NSWEEPS: 10
BATCH_SIZE: 5 # 5 for 2080ti, 15 for v100
```
Specify Tasks
```
 HEAD:
    TASKS:
        - {num_class: 1, class_names: ["car"]}
        - {num_class: 2, class_names: ["truck", "construction_vehicle"]}
        - {num_class: 2, class_names: ["bus", "trailer"]}
        - {num_class: 1, class_names: ["barrier"]}
        - {num_class: 2, class_names: ["motorcycle", "bicycle"]}
        - {num_class: 2, class_names: ["pedestrian", "traffic_cone"]}
```

Run
------------
For better experiments organization, I suggest the following scripts:
```
./tools/scripts/train.sh
```

## Benchmark

|              | KITTI(Val) | nuScenes(Val) |
| ------------ | ---------- | ------------- |
| VoxelNet     | √          | √             |
| SECOND       | √          | √             |
| PointPillars | √          | √             |
| PIXOR        | √          | √             |
| PointRCNN    | x          | x             |
| CBGS         | √          | √             |
| ViP          | x          | x             |

## 4. Currently Support

* Models
  - [x] VoxelNet
  - [x] SECOND
  - [x] PointtPillars
  - [x] PIXOR
  - [x] SENet & GCNet (GCNet will course model output 0, deprecated.)
  - [x] Pointnet++
  - [x] EKF Tracker & IoU Tracker
  - [x] PointRCNN

* Features
  - [x] Multi task learning
  - [x] Single-gpu & Distributed Training and Validation
  - [x] GradNorm for Multi-task Training
  - [x] Flexible anchor dimensions
  - [x] TensorboardX
  - [x] Checkpointer & breakpoint continue
  - [x] Support both KITTI and nuScenes Dataset
  - [x] SyncBN
  - [x] Self-contained visualization
  - [x] YAML configuration
  - [x] Finetune
  - [x] Multiscale Training & Validation
  - [x] Rotated RoI Align


## 5. TODO List
* Models
  - [ ] FrustumPointnet
  - [ ] VoteNet

* Features


