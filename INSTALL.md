##Installation

### Requirements

- Linux
- Python 3.6+
- PyTorch 1.1 or higher
- CUDA 10.0 or higher
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

###Install Det3D

a. Clone the Det3D repository

```bash
git clone https://github.com/poodarchu/Det3D.git
cd Det3D
```

b. Install APEX and spconv following the official instructions.

c. Install Det3D

```
pip install -r requirements.txt
python setup.py build develop
```

