## Installation

### Requirements

- Linux
- Python 3.6+
- PyTorch 1.1-1.6
- CUDA 10.0/10.1
- **CMake 3.13.2 or higher**
- [spconv](https://github.com/poodarchu/spconv) 
- [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

**spconv and nuscenes-devkit should be the specific version from link above**

we have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04
- Python: 3.6.5
- PyTorch: 1.1
- CUDA: 10.0
- CUDNN: 7.5.0


### Install Requirements

#### spconv

```bash
 $ sudo apt-get install libboost-all-dev
 $ git clone https://github.com/poodarchu/spconv --recursive
 $ cd spconv && python setup.py bdist_wheel
 $ cd ./dist && pip install *
```

#### nuscenes-devkit

```bash
pip install nuscenes-devkit
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

#### cannot import name PILLOW_VERSION
`pip install Pillow==6.1`

#### Installing a suitable pytorch version by replacing the previous version
`pip install torch==1.3.0 torchvision==0.4.1`

#### Upgrading cmake in case if needed
`sudo apt remove cmake`

`pip install cmake --upgrade`

#### Installing suitable setuptools version by replacing the previous version
`pip install setuptools==39.1.0`

