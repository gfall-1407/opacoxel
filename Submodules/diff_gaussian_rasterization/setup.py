from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="diff_gaussian_render",
    packages=['diff_gaussian_render'],
    ext_modules=[
        CppExtension(
            name="diff_gaussian_render._C",
            sources=[
            "ext.cpp"],
            )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)