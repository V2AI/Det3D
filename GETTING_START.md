## Getting Started with Det3D

This document aims to provide a brief introduction to use Det3D

### Prepare data

####1.download data and organise as follows

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
       
 # For Lyft Dataset
 └── LYFT_DATASET_ROOT
       ├── trainval 
       |   ├── data
       |   ├── lidar
       |   └── maps
       └── test
           ├── data
           ├── lidar
           └── maps
```
####2.create data 

```
# KITTI
python tools/create_data.py kitti_data_prep --root_path=KITTI_DATASET_ROOT

# nuScenes
python tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10

# Lyft
python tools/create_data.py lyft_data_prep --root_path=LYFT_DATASET_ROOT
```

### Train & Evaluate in Command Line

The scripts used to train and evaluate is located in  `./tools/scripts/`. 

To train a model with `./tools/scripts/train.sh`, first setup **nproc_per_node**, **config_path ** and **work_dir** appropriately. Then run:

`./tools/sctrips/train.sh TASK_DESCRIPTION `

To test a model with `./tools/scripts/test.sh`, run:

`./tools/scripts/test.sh CONFIG_PATH WORK_DIR CHECKPOINT `

Where `CHECKPOINT` represents the path of model file.

