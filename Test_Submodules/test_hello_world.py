import torch  # 先确保PyTorch正常  
import hello_world_extension # 直接导入C++模块

hello_world_extension.hello_world()  # 调用C++函数