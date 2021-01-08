## Getting Started with Det3D

This document aims to provide a brief introduction to use Det3D

### Prepare data

#### Download data and organise as follows

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
#### Create data

Data creation should be under the gpu environment.

```
# KITTI
python tools/create_data.py kitti_data_prep --root_path=KITTI_DATASET_ROOT

# nuScenes
python tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10

# Lyft
python tools/create_data.py lyft_data_prep --root_path=LYFT_DATASET_ROOT
```

### Modify Configs

#### Update dataset setting and path

```python
dataset_type = "NuScenesDataset"
n_sweeps = 10
data_root = "/data/Datasets/nuScenes"
db_info_path="/data/Datasets/nuScenes/dbinfos_train.pkl"
train_anno = "/data/Datasets/nuScenes/infos_train_10sweeps_withvelo.pkl"
val_anno = "/data/Datasets/nuScenes/infos_val_10sweeps_withvelo.pkl"
```

#### Specify Task and Anchor

**The order of tasks and anchors must be the same**

```python
tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]
anchor_generators=[
    dict(
        type="anchor_generator_range",
        sizes=[1.97, 4.63, 1.74],
        anchor_ranges=[-50.4, -50.4, -0.95, 50.4, 50.4, -0.95],
        rotations=[0, 1.57],
        velocities=[0, 0],
        matched_threshold=0.6,
        unmatched_threshold=0.45,
        class_name="car",
    ),
    dict(
        type="anchor_generator_range",
        sizes=[2.51, 6.93, 2.84],
        anchor_ranges=[-50.4, -50.4, -0.40, 50.4, 50.4, -0.40],
        rotations=[0, 1.57],
        velocities=[0, 0],
        matched_threshold=0.55,
        unmatched_threshold=0.4,
        class_name="truck",
    ),
    ...
]
```



### Train & Evaluate in Command Line

**Now we only support training and evaluation with gpu. Cpu only mode is not supported.**

The scripts used to train and evaluate is located in  `./tools/scripts/`. 

To train a model with `./tools/scripts/train.sh`, first setup **nproc_per_node**, **config_path ** and **work_dir** appropriately. Then run:

`./tools/sctrips/train.sh TASK_DESCRIPTION `

To test a model with `./tools/scripts/test.sh`, run:

`./tools/scripts/test.sh CONFIG_PATH WORK_DIR CHECKPOINT `

Where `CHECKPOINT` represents the path of model file.

### Common Issues

* `qt.qpa.screen: QXcbConnection: Could not connect to display localhost:11.0`
  - SOLUTION: ```export QT_QPA_PLATFORM='offscreen'``` 
