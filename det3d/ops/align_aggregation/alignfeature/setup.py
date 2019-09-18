from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="align_feature",
    ext_modules=[
        CUDAExtension(
            "align_feature_cuda",
            ["src/align_feature.cpp", "src/align_feature_cuda_kernel.cu",],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
