## Installation

### Requirements

- Linux
- Python 3.6+
- PyTorch 1.1 or higher
- CUDA 10.0 or higher
- CMake 3.13.2 or higher
- [APEX](https://github.com/nvidia/apex)
- [spconv](https://github.com/traveller59/spconv/commit/73427720a539caf9a44ec58abe3af7aa9ddb8e39) 
- [nuscenes-devkit](https://github.com/poodarchu/nuscenes/)

**spconv and nuscenes-devkit should be the specific version from link above**

we have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04
- Python: 3.6.5
- PyTorch: 1.1
- CUDA: 10.0
- CUDNN: 7.5.0

### Install Requirements

Installation of APEX and spconv should be unver the gpu environment.

#### APEX

```bash
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### spconv

```bash
 $ sudo apt-get install libboost-all-dev
 $ git clone https://github.com/traveller59/spconv.git --recursive
 $ cd spconv && git checkout 7342772
 $ python setup.py bdist_wheel
 $ cd ./dist && pip install *
```

#### nuscenes-devkit

```bash
$ git clone https://github.com/poodarchu/nuscenes.git
$ cd nuscenes
$ python setup.py install
```

### Install Det3D

The installation should be under the gpu environment.

#### Clone the Det3D repository

```bash
$ git clone https://github.com/poodarchu/Det3D.git
$ cd Det3D
```

#### Install Det3D

```bash
$ python setup.py build develop
```

### Common Installation Issues

#### ModuleNotFoundError: No module named 'det3d.ops.nms.nms' when installing det3d

Run `python setup.py build develop` again.

#### "values of 'package_data' dict" must be a list of strings (got '*.json') when installing nuscenes-devikit

Use `setuptools 39.1.0 `

