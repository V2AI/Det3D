from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="correlation_cuda",
    ext_modules=[
        CUDAExtension(
            "correlation_cuda",
            ["src/correlation.cpp", "src/correlation_cuda_kernel.cu",],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
