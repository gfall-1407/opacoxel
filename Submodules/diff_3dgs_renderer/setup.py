from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    name="diff_3dgs_renderer",
    packages=['diff_3dgs_renderer'],
    ext_modules=[
        CUDAExtension(
            name="diff_3dgs_renderer._C",
            sources=[
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)