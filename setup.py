import copy
import glob
import os
import platform
import subprocess
import time
from collections import defaultdict

from setuptools import Extension, dist, find_packages, setup

import numpy as np  # noqa: E402
from Cython.Build import cythonize  # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

dist.Distribution().fetch_build_eggs(["Cython", "numpy>=1.11.1"])


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


MAJOR = 1
MINOR = 0
PATCH = ""
SUFFIX = "rc0"
SHORT_VERSION = "{}.{}.{}{}".format(MAJOR, MINOR, PATCH, SUFFIX)

version_file = "det3d/version.py"


def get_git_hash():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH", "HOME"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        sha = out.strip().decode("ascii")
    except OSError:
        sha = "unknown"

    return sha


def get_hash():
    if os.path.exists(".git"):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from det3d.version import __version__

            sha = __version__.split("+")[-1]
        except ImportError:
            raise ImportError("Unable to get git version")
    else:
        sha = "unknown"

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
short_version = '{}'
"""
    sha = get_hash()
    VERSION = SHORT_VERSION + "+" + sha

    with open(version_file, "w") as f:
        f.write(content.format(time.asctime(), VERSION, SHORT_VERSION))


def get_version():
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


def make_cuda_ext(name, module, sources, extra_compile_args={}):

    if "cxx" not in extra_compile_args:
        extra_compile_args["cxx"] = []

    nvcc_flags = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    if "nvcc" not in extra_compile_args:
        extra_compile_args["nvcc"] = nvcc_flags
    else:
        extra_compile_args["nvcc"] += nvcc_flags

    return CUDAExtension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        extra_compile_args=copy.deepcopy(extra_compile_args),
    )


def make_cython_ext(name, module, sources, extra_compile_args={}):
    if platform.system() != "Windows":
        extra_compile_args["cxx"] += ["-Wno-unused-function", "-Wno-write-strings"]

    extension = Extension(
        "{}.{}".format(module, name),
        [os.path.join(*module.split("."), p) for p in sources],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
    return extension


def get_requirements(filename="requirements.txt"):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), "r") as f:
        requires = [line.replace("\n", "") for line in f.readlines()]
    return requires


if __name__ == "__main__":
    write_version_py()
    setup(
        name="det3d",
        version=get_version(),
        description="det3d: A General 3D Object Detection Codebase",
        long_description=readme(),
        author="Benjin Zhu",
        author_email="poodarchu@gmail.com",
        keywords="computer vision, 3d object detection",
        url="https://github.com/poodarchu/det3d",
        packages=find_packages(exclude=("configs", "tools", "demo")),
        package_data={"det3d.ops": ["*/*.so"]},
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
        ],
        license="Apache License 2.0",
        setup_requires=["pytest-runner", "cython", "numpy"],
        tests_require=["pytest"],
        install_requires=get_requirements(),
        ext_modules=[
            CUDAExtension(
                name="det3d.ops.pointnet2.PN2",
                sources=glob.glob("det3d/ops/pointnet2/src/*.cpp")
                + glob.glob("det3d/ops/pointnet2/src/*.cu"),
                extra_compile_args={"cxx": ["-g",], "nvcc": ["-g",],},
            ),
            CUDAExtension(
                name="det3d.ops.rroi_align.RotateRoIAlign",
                sources=[
                    "det3d/ops/rroi_align/ROIAlign_cuda.cpp",
                    "det3d/ops/rroi_align/ROIAlign_cuda_kernel.cu",
                ],
            ),
            CUDAExtension(
                name="det3d.ops.roipool3d.RoIPool3D",
                sources=[
                    "det3d/ops/roipool3d/src/roipool3d.cpp",
                    "det3d/ops/roipool3d/src/roipool3d_kernel.cu",
                ],
                extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
            ),
            CUDAExtension(
                name="det3d.ops.iou3d.IoU3D",
                sources=[
                    "det3d/ops/iou3d/src/iou3d.cpp",
                    "det3d/ops/iou3d/src/iou3d_kernel.cu",
                ],
                extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
            ),
            CUDAExtension(
                name="det3d.ops.nms.nms",
                sources=[
                    "det3d/ops/nms/nms.cc",
                    "det3d/ops/nms/nms_kernel.cu",
                ],
                extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
            ),
            make_cuda_ext(
                name="sigmoid_focal_loss_cuda",
                module="det3d.ops.sigmoid_focal_loss",
                sources=[
                    "src/sigmoid_focal_loss.cpp",
                    "src/sigmoid_focal_loss_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="syncbn_gpu",
                module="det3d.ops.syncbn",
                sources=["src/syncbn_cuda.cpp", "src/syncbn_cuda_kernel.cu"],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
