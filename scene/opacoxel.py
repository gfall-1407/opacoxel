import torch  
import torch.nn as nn  
import numpy as np  
from typing import Tuple, Optional  
try:
    from skimage import measure as sk_measure
    _SKIMAGE_OK = True
except Exception:
    _SKIMAGE_OK = False
  
class Opacoxels:  
    def __init__(self,   
                 bounds: Tuple[float, float, float, float, float, float] = (-1, 1, -1, 1, -1, 1),  
                 resolution: Tuple[int, int, int] = (256, 256, 256),  
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

        # 先对坐标逐维裁剪，避免后续 ceil 溢出到网格外
        max_idx_f = torch.tensor([nx - 1, ny - 1, nz - 1], device=self.device, dtype=coords.dtype)
        min_idx_f = torch.zeros(3, device=self.device, dtype=coords.dtype)
        coords = torch.clamp(coords, min=min_idx_f, max=max_idx_f - 1e-6)

        # 获取整数和小数部分
        coords_floor = torch.floor(coords).long()
        coords_frac = coords - coords_floor.float()

        # 边界检查（逐维张量上下界）
        max_idx_l = max_idx_f.long()
        min_idx_l = torch.zeros(3, device=self.device, dtype=coords_floor.dtype)
        coords_floor = torch.clamp(coords_floor, min=min_idx_l, max=max_idx_l)
        coords_ceil = torch.clamp(coords_floor + 1, min=min_idx_l, max=max_idx_l)
          
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

    def update_from_points(self, positions: torch.Tensor, opacities: torch.Tensor):
        """
        直接由点与其不透明度更新体素网格（无需 Gaussians 包装）。

        Args:
            positions: 世界坐标 [N, 3]
            opacities: 不透明度 [N] 或 [N,1]
        """
        if positions.device.type != self.device:
            positions = positions.to(self.device)
        if opacities.dim() == 2 and opacities.shape[1] == 1:
            opacities = opacities.squeeze(1)
        if opacities.device.type != self.device:
            opacities = opacities.to(self.device)

        voxel_coords = self.world_to_voxel(positions)
        self._distribute_to_voxels(voxel_coords, opacities)
      
    def _distribute_to_voxels(self, voxel_coords: torch.Tensor, opacities: torch.Tensor):  
        """  
        将opacity值分布到体素网格中  
          
        Args:  
            voxel_coords: 体素坐标 [N, 3]  
            opacities: opacity值 [N]  
        """  
        # 近邻体素索引
        voxel_indices = torch.round(voxel_coords).long()

        # 边界裁剪
        nx, ny, nz = self.resolution
        valid_mask = (
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < nx) &
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < ny) &
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < nz)
        )
        if valid_mask.sum() == 0:
            return

        voxel_indices = voxel_indices[valid_mask]
        valid_opacities = opacities[valid_mask]

        # 将三维索引展平为一维索引，便于 scatter
        flat_indices = (
            voxel_indices[:, 0] * (ny * nz) +
            voxel_indices[:, 1] * nz +
            voxel_indices[:, 2]
        )

        # 准备聚合缓冲区：sum 和 count，用于平均（比逐点 max 更稳健）
        flat_size = nx * ny * nz
        sum_buffer = torch.zeros(flat_size, device=self.device, dtype=self.opacity_grid.dtype)
        cnt_buffer = torch.zeros_like(sum_buffer)

        sum_buffer.index_add_(0, flat_indices, valid_opacities)
        cnt_buffer.index_add_(0, flat_indices, torch.ones_like(valid_opacities))

        # 计算平均并写回
        avg_buffer = torch.zeros_like(sum_buffer)
        nonzero = cnt_buffer > 0
        avg_buffer[nonzero] = sum_buffer[nonzero] / cnt_buffer[nonzero]

        avg_grid = avg_buffer.view(nx, ny, nz)
        # 融合策略：取较大值，避免过度抹平
        self.opacity_grid = torch.maximum(self.opacity_grid, avg_grid)
      
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

    def extract_surface(self, iso_level: float = 0.5, allow_degenerate: bool = True):
        """
        使用 Marching Cubes 从不透明度体素中提取等值面（世界坐标）。

        Args:
            iso_level: 提取阈值（不透明度等值）
            allow_degenerate: 是否允许退化三角形

        Returns:
            vertices_world: (V,3) numpy 数组，世界坐标
            faces: (F,3) numpy int32 数组
            normals_world: (V,3) numpy 数组，世界坐标方向单位法线
            values: (V,) numpy 数组，对应顶点的体素值
        """
        if not _SKIMAGE_OK:
            raise ImportError("scikit-image 未安装，无法执行 marching cubes。请先安装: pip install scikit-image")

        # skimage 假定体素排列为 (Z, Y, X)。当前 grid 为 (X, Y, Z)，需转置。
        grid_np = self.opacity_grid.detach().clone().clamp(0.0, 1.0).cpu().numpy().transpose(2, 1, 0)

        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        nx, ny, nz = self.resolution

        # 体素间距（按索引坐标 -> 世界坐标的缩放）
        dx = (x_max - x_min) / max(nx - 1, 1)
        dy = (y_max - y_min) / max(ny - 1, 1)
        dz = (z_max - z_min) / max(nz - 1, 1)

        # 运行 marching cubes
        verts, faces, normals, values = sk_measure.marching_cubes(
            volume=grid_np,
            level=iso_level,
            spacing=(dz, dy, dx),  # 注意顺序与 (Z, Y, X) 对齐
            allow_degenerate=allow_degenerate
        )

        # verts 当前在体素索引空间原点 (0,0,0) 对应世界 (z_min, y_min, x_min)，需平移到世界坐标
        # verts 顺序为 (z, y, x)
        x_world = x_min + verts[:, 2]
        y_world = y_min + verts[:, 1]
        z_world = z_min + verts[:, 0]
        vertices_world = np.stack([x_world, y_world, z_world], axis=1)

        # 法线按相同轴重排（已在 spacing 中缩放，无需再缩放，只需重排到 (x,y,z)）
        normals_world = np.stack([normals[:, 2], normals[:, 1], normals[:, 0]], axis=1)

        return vertices_world.astype(np.float32), faces.astype(np.int32), normals_world.astype(np.float32), values.astype(np.float32)