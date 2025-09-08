import torch  
import torch.nn as nn  
import numpy as np  
from typing import Tuple, Optional  
  
class Opacoxel:  
    def __init__(self,   
                 bounds: Tuple[float, float, float, float, float, float] = (-1, 1, -1, 1, -1, 1),  
                 resolution: Tuple[int, int, int] = (64, 64, 64),  
                 device: str = "cuda"):  
        """  
        初始化体素化opacity field  
          
        Args:  
            bounds: 3D空间边界 (x_min, x_max, y_min, y_max, z_min, z_max)  
            resolution: 体素网格分辨率 (nx, ny, nz)  
            device: 计算设备  
        """  
        self.bounds = bounds  
        self.resolution = resolution  
        self.device = device  
          
        # 计算体素大小  
        self.voxel_size = (  
            (bounds[1] - bounds[0]) / resolution[0],  
            (bounds[3] - bounds[2]) / resolution[1],   
            (bounds[5] - bounds[4]) / resolution[2]  
        )  
          
        # 初始化opacity体素网格  
        self.opacity_grid = torch.zeros(resolution, device=device, dtype=torch.float32)  
          
    def world_to_voxel(self, positions: torch.Tensor) -> torch.Tensor:  
        """  
        将世界坐标转换为体素坐标  
          
        Args:  
            positions: 世界坐标 [N, 3]  
              
        Returns:  
            体素坐标 [N, 3]  
        """  
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds  
        nx, ny, nz = self.resolution  
          
        # 归一化到[0, 1]  
        normalized = torch.stack([  
            (positions[:, 0] - x_min) / (x_max - x_min),  
            (positions[:, 1] - y_min) / (y_max - y_min),   
            (positions[:, 2] - z_min) / (z_max - z_min)  
        ], dim=1)  
          
        # 转换为体素索引  
        voxel_coords = torch.stack([  
            normalized[:, 0] * (nx - 1),  
            normalized[:, 1] * (ny - 1),  
            normalized[:, 2] * (nz - 1)  
        ], dim=1)  
          
        return voxel_coords  
      
    def sample_opacity(self, positions: torch.Tensor) -> torch.Tensor:  
        """  
        在给定位置采样opacity值  
          
        Args:  
            positions: 查询位置 [N, 3]  
              
        Returns:  
            opacity值 [N]  
        """  
        voxel_coords = self.world_to_voxel(positions)  
          
        # 使用三线性插值  
        return self._trilinear_interpolation(voxel_coords)  
      
    def _trilinear_interpolation(self, coords: torch.Tensor) -> torch.Tensor:  
        """  
        三线性插值采样  
          
        Args:  
            coords: 体素坐标 [N, 3]  
              
        Returns:  
            插值后的opacity值 [N]  
        """  
        nx, ny, nz = self.resolution  
          
        # 获取整数和小数部分  
        coords_floor = torch.floor(coords).long()  
        coords_frac = coords - coords_floor.float()  
          
        # 边界检查  
        coords_floor = torch.clamp(coords_floor, 0, torch.tensor([nx-1, ny-1, nz-1], device=self.device))  
        coords_ceil = torch.clamp(coords_floor + 1, 0, torch.tensor([nx-1, ny-1, nz-1], device=self.device))  
          
        # 获取8个邻近体素的opacity值  
        x0, y0, z0 = coords_floor[:, 0], coords_floor[:, 1], coords_floor[:, 2]  
        x1, y1, z1 = coords_ceil[:, 0], coords_ceil[:, 1], coords_ceil[:, 2]  
          
        # 8个角点的值  
        c000 = self.opacity_grid[x0, y0, z0]  
        c001 = self.opacity_grid[x0, y0, z1]  
        c010 = self.opacity_grid[x0, y1, z0]  
        c011 = self.opacity_grid[x0, y1, z1]  
        c100 = self.opacity_grid[x1, y0, z0]  
        c101 = self.opacity_grid[x1, y0, z1]  
        c110 = self.opacity_grid[x1, y1, z0]  
        c111 = self.opacity_grid[x1, y1, z1]  
          
        # 三线性插值  
        xd, yd, zd = coords_frac[:, 0], coords_frac[:, 1], coords_frac[:, 2]  
          
        c00 = c000 * (1 - xd) + c100 * xd  
        c01 = c001 * (1 - xd) + c101 * xd  
        c10 = c010 * (1 - xd) + c110 * xd  
        c11 = c011 * (1 - xd) + c111 * xd  
          
        c0 = c00 * (1 - yd) + c10 * yd  
        c1 = c01 * (1 - yd) + c11 * yd  
          
        result = c0 * (1 - zd) + c1 * zd  
          
        return result  
      
    def update_from_gaussians(self, gaussians):  
        """  
        从Gaussians对象更新体素网格  
          
        Args:  
            gaussians: Gaussians对象实例  
        """  
        # 获取Gaussian的位置和opacity  
        positions = gaussians.get_position  # [N, 3]  
        opacities = gaussians.get_opacity.squeeze()  # [N]  
          
        # 将Gaussian的opacity分布到体素网格中  
        voxel_coords = self.world_to_voxel(positions)  
          
        # 使用最近邻或加权平均方式更新体素  
        self._distribute_to_voxels(voxel_coords, opacities)  
      
    def _distribute_to_voxels(self, voxel_coords: torch.Tensor, opacities: torch.Tensor):  
        """  
        将opacity值分布到体素网格中  
          
        Args:  
            voxel_coords: 体素坐标 [N, 3]  
            opacities: opacity值 [N]  
        """  
        # 四舍五入到最近的体素  
        voxel_indices = torch.round(voxel_coords).long()  
          
        # 边界检查  
        nx, ny, nz = self.resolution  
        valid_mask = (  
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < nx) &  
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < ny) &  
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < nz)  
        )  
          
        valid_indices = voxel_indices[valid_mask]  
        valid_opacities = opacities[valid_mask]  
          
        # 更新体素网格（使用最大值或平均值）  
        for i in range(valid_indices.shape[0]):  
            x, y, z = valid_indices[i]  
            self.opacity_grid[x, y, z] = torch.max(  
                self.opacity_grid[x, y, z],   
                valid_opacities[i]  
            )  
      
    def get_opacity_at_position(self, position: torch.Tensor) -> float:  
        """  
        获取指定位置的opacity值  
          
        Args:  
            position: 3D位置 [3]  
              
        Returns:  
            opacity值  
        """  
        if position.dim() == 1:  
            position = position.unsqueeze(0)  
          
        opacity = self.sample_opacity(position)  
        return opacity.item() if opacity.numel() == 1 else opacity  
      
    def clear(self):  
        """清空体素网格"""  
        self.opacity_grid.zero_()  
      
    def save_to_file(self, filepath: str):  
        """保存体素网格到文件"""  
        torch.save({  
            'opacity_grid': self.opacity_grid.cpu(),  
            'bounds': self.bounds,  
            'resolution': self.resolution,  
            'voxel_size': self.voxel_size  
        }, filepath)  
      
    def load_from_file(self, filepath: str):  
        """从文件加载体素网格"""  
        data = torch.load(filepath, map_location=self.device)  
        self.opacity_grid = data['opacity_grid'].to(self.device)  
        self.bounds = data['bounds']  
        self.resolution = data['resolution']  
        self.voxel_size = data['voxel_size']