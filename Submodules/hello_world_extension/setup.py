from setuptools import setup  
from torch.utils.cpp_extension import CppExtension, BuildExtension  
  
setup(  
    name="hello_world_extension",  
    packages=['hello_world_extension'],  
    ext_modules=[  
        CppExtension(  
            name="hello_world_extension._C",  
            sources=["ext.cpp"]  
        )  
    ],  
    cmdclass={'build_ext': BuildExtension}  
)