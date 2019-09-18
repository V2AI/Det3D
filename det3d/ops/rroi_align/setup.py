# -*- coding:utf-8 -*-
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="RotateRoIAlign",
    version="0.1",
    ext_modules=[
        CUDAExtension(
            "RotateRoIAlign_cuda", ["ROIAlign_cuda.cpp", "ROIAlign_cuda_kernel.cu",]
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
