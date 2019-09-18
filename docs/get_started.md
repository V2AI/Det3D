# 3D Detection Get Started

## 1. 激光雷达和点云

关于激光雷达的原理，可以参考下面几篇文章：

* https://zhuanlan.zhihu.com/p/33792450
* https://pdal.io/workshop/lidar-introduction.html

## 2. 数据集

* **PASCAL3D+ (2014)** [[Link\]](http://cvgl.stanford.edu/projects/pascal3d.html) 
  12 categories, on average 3k+ objects per category, for 3D object detection and pose estimation.
* **ModelNet (2015)** [[Link\]](http://modelnet.cs.princeton.edu/#) 
  127915 3D CAD models from 662 categories 
  ModelNet10: 4899 models from 10 categories 
  ModelNet40: 12311 models from 40 categories, all are uniformly orientated
* **ShapeNet (2015)** [[Link\]](https://www.shapenet.org/) 
  3Million+ models and 4K+ categories. A dataset that is large in scale, well organized and richly annotated. 
  ShapeNetCore [[Link\]](http://shapenet.cs.stanford.edu/shrec16/): 51300 models for 55 categories.
* **NYU Depth Dataset V2 (2012)** [[Link\]](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) 
  1449 densely labeled pairs of aligned RGB and depth images from Kinect video sequences for a variety of indoor scenes.
* **SUNRGB-D 3D Object Detection Challenge** [[Link\]](http://rgbd.cs.princeton.edu/challenge.html) 
  19 object categories for predicting a 3D bounding box in real world dimension 
  Training set: 10,355 RGB-D scene images, Testing set: 2860 RGB-D images
* **ScanNet (2017)** [[Link\]](http://www.scan-net.org/) 
  An RGB-D video dataset containing 2.5 million views in more than 1500 scans, annotated with 3D camera poses, surface reconstructions, and instance-level semantic segmentations.
* **Facebook House3D: A Rich and Realistic 3D Environment (2017)** [[Link\]](https://github.com/facebookresearch/House3D) 
  House3D is a virtual 3D environment which consists of 45K indoor scenes equipped with a diverse set of scene types, layouts and objects sourced from the SUNCG dataset. All 3D objects are fully annotated with category labels. Agents in the environment have access to observations of multiple modalities, including RGB images, depth, segmentation masks and top-down 2D map views.

### 2.1自动驾驶数据集

* ### KITTI Benckmark
    [paper link](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf)
    The [KITTI](http://www.cvlibs.net/datasets/kitti/) (**K**arlsruhe **I**nstitute of **T**echnology and **T**oyota Technological **I**nstitute) dataset is a widely used computer vision benchmark which was released in 2012. A Volkswagen station was fitted with grayscale and color cameras, a Velodyne 3D Laser Scanner and a GPS/IMU system. They have datasets for various scenarios like urban, [residential](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=residential), [highway](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=road), and [campus](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=campus). 
  
* ### nuScenes Benckmark
    [paper link](https://arxiv.org/abs/1903.11027v1)
    nuTonomy scenes (nuScenes) is the first dataset to carry the full autonomous vehicle sensor suite: 6 cameras, 5 radars and 1 lidar, all with full 360 degree field ofview. nuScenes comprises 1000 scenes, each 20s long and fully annotated with 3D bound- ing boxes for 23 classes and 8 attributes. It has 7x as many annotations and 100x as many images as the pioneering KITTI dataset.


## 3. Papers List

### 3.1 Voxel-based Methods

* [Second: Sparsely embedded convolutional detection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6210968/)
* [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/abs/1711.06396)
* [PIXOR: Real-time 3D Object Detection from Point Clouds](https://arxiv.org/abs/1902.06326v3)
* [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)

### 3.2 Point-based Methods

* [PointNet: Deep learning on point sets for 3D classification and segmentation](http://arxiv.org/abs/1612.00593)
* [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](http://arxiv.org/abs/1706.02413)
* [PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation](http://arxiv.org/abs/1807.00652)
* [SO-Net: Self-Organizing Network for Point Cloud Analysis](arXiv:1803.04249v4)
* [PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud](http://arxiv.org/abs/1812.04244)
* [IPOD: Intensive Point-based Object Detector for Point Cloud](http://arxiv.org/abs/1812.05276)
* [Deep Hough Voting for 3D Object Detection in Point Clouds](http://arxiv.org/abs/1904.09664)

### 3.3 BEV & Multi-View

* [YOLO3D](https://arxiv.org/pdf/1808.02350v1.pdf)
* [Complex-YOLO: An Euler-Region-Proposal for
  Real-time 3D Object Detection on Point Clouds](https://arxiv.org/abs/1803.06199v2)
* [Complexer-YOLO: Real-Time 3D Object Detection and Tracking on Semantic Point Clouds](https://arxiv.org/abs/1904.07537v1)
* [Joint 3D Proposal Generation and Object Detection from View Aggregation](https://arxiv.org/abs/1712.02294v4)
* [Multi-View 3D Object Detection Network for Autonomous Driving](https://arxiv.org/abs/1611.07759)

### 3.4 Depth & Monocular

### 3.5 Sensor Fusion & Tracking



## 4. Libraries

### 点云处理

* PCL
* Open3D (推荐)



## 5. Blogs

* 之前的一次 3D 分享： https://zhuanlan.zhihu.com/p/58734240



