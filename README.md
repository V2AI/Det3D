# Det3D

A general 3D Object Detection codebase in PyTorch

## Introduction

Det3D is the first 3D Object Detection toolbox which provides off the box implementations of many 3D object detection algorithms such as PointPillars, SECOND, PIXOR, etc, as well as state-of-the-art methods on major benchmarks like KITTI(ViP) and nuScenes(CBGS). Key features of Det3D include the following aspects:

* Multi Datasets Support: KITTI, nuScenes, Lyft
* Point-based and Voxel-based model zoo
* State-of-the-art performance
* DDP & SyncBN

## Installation

Please refer to [INSTALL.md](INSTALL.md).

## Quick Start

Please refer to [GETTING_START.md](GETTING_START.md).

## Model Zoo and Baselines

We provide many baseline results and trained models. Please refer to [MODEL_ZOO.md](MODEL_ZOO.md).

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

Det3D is released under the MIT license.

## Acknowlegement

* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```BIB
@article{CBGS,
  author    = {Benjin Zhu and
               Zhengkai Jiang and
               Xiangxin Zhou and
               Zeming Li and
               Gang Yu},
  title     = {Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection},
  journal   = {arXiv preprint arXiv:1908.094925},
  year      = {2019},
}
```