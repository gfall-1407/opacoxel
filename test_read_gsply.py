import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement
try:
    import open3d as o3d
    _O3D_OK = True
except Exception:
    _O3D_OK = False
import os
import torch
from scene import Opacoxels
from utils.sh_utils import sh_to_RGB

class GaussianPLY(NamedTuple):
    positions : np.array
    scales : np.array
    rotations : np.array 
    features_dc : np.array
    features_rest : np.array
    opacities : np.array

def read_gs_ply(path, max_sh_degree):
    plydata = PlyData.read(path)
    positions = np.stack((np.asarray(plydata.elements[0]["x"]),
          np.asarray(plydata.elements[0]["y"]),
          np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    
    features_dc = np.zeros((positions.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_rest = np.zeros((positions.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_rest = features_rest.reshape((features_rest.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((positions.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((positions.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    return GaussianPLY(positions=positions,
                       scales=scales,
                       rotations=rots,
                       features_dc=features_dc,
                       features_rest=features_rest,
                       opacities=opacities)

def compute_bbox(positions: np.array):
    x_np = positions[:,0]
    y_np = positions[:,1]
    z_np = positions[:,2]
    x_min, x_max = x_np.min(), x_np.max()
    y_min, y_max = y_np.min(), y_np.max()
    z_min, z_max = z_np.min(), z_np.max()
    x_len = x_max - x_min
    y_len = y_max - y_min
    z_len = z_max - z_min
    radius = max(max(x_len, y_len), z_len)
    bbox = [x_min, x_max, y_min, y_max, z_min, z_max, radius*1.1]
    return bbox

if __name__ == "__main__":
    gaussians = read_gs_ply("PLYs/point_cloud_30000.ply", 3)
    # 构造 PLY 顶点与面
    save_dir = "PLYs/output"
    vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    verts = gaussians.positions
    # 将读取的 Gaussians 保存为点云：xyz=positions，normal=0，rgb=sh_to_RGB(features_dc)
    pc_save_path = os.path.join(save_dir, "points_from_gs_ply.ply")
    num_pts = gaussians.positions.shape[0]
    # positions
    x = gaussians.positions[:, 0].astype(np.float32)
    y = gaussians.positions[:, 1].astype(np.float32)
    z = gaussians.positions[:, 2].astype(np.float32)
    # normals 全 0
    nx = np.zeros_like(x, dtype=np.float32)
    ny = np.zeros_like(y, dtype=np.float32)
    nz = np.zeros_like(z, dtype=np.float32)
    # colors 由 DC SH 转 RGB
    # features_dc: (N, 3, 1) -> (N, 3)
    dc = gaussians.features_dc.squeeze(-1)
    dc_t = torch.from_numpy(dc).float()
    rgb01 = sh_to_RGB(dc_t).clamp(0.0, 1.0).cpu().numpy()
    r = (rgb01[:, 0] * 255.0).astype(np.uint8)
    g = (rgb01[:, 1] * 255.0).astype(np.uint8)
    b = (rgb01[:, 2] * 255.0).astype(np.uint8)

    vertex_dtype_pc = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                       ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                       ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertex_data_pc = np.empty(num_pts, dtype=vertex_dtype_pc)
    vertex_data_pc["x"], vertex_data_pc["y"], vertex_data_pc["z"] = x, y, z
    vertex_data_pc["nx"], vertex_data_pc["ny"], vertex_data_pc["nz"] = nx, ny, nz
    vertex_data_pc["red"], vertex_data_pc["green"], vertex_data_pc["blue"] = r, g, b
    PlyData([PlyElement.describe(vertex_data_pc, "vertex")], text=True).write(pc_save_path)
    print(f"Saved point cloud to {pc_save_path} with {num_pts} points.")

    # 使用点云包围盒初始化体素边界，并设置体素分辨率
    bbox = compute_bbox(gaussians.positions)
    x_min, x_max, y_min, y_max, z_min, z_max, _ = bbox
    bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
    resolution = (512, 512, 512)
    opacoxels = Opacoxels(bounds=bounds, resolution=resolution, device="cuda" if torch.cuda.is_available() else "cpu")

    # 快速方法：将点云体素化为占据体，再直接 Marching Cubes（更快）
    print("Running fast voxel-based surface extraction from positions...")
    # 使用更低分辨率加速（可按需调高），并将所有点 opacity 置 1
    fast_res = (512, 512, 512)
    opacoxels_fast = Opacoxels(bounds=bounds, resolution=fast_res, device="cpu")
    #ones_opacity = torch.ones((gaussians.positions.shape[0], 1), device=opacoxels_fast.device, dtype=torch.float32)
    #opacoxels_fast.update_from_points(torch.from_numpy(gaussians.positions).float().to(opacoxels_fast.device), ones_opacity)
    fast_iso = 0.5
    f_verts, f_faces, f_norms, _ = opacoxels_fast.extract_surface(iso_level=fast_iso)

    #fast_mesh_path = os.path.join(save_dir, "fast_voxel_mesh.ply")
    #f_vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    #f_vertex_data = np.empty(f_verts.shape[0], dtype=f_vertex_dtype)
    #f_vertex_data["x"], f_vertex_data["y"], f_vertex_data["z"] = f_verts[:, 0], f_verts[:, 1], f_verts[:, 2]
    #f_vertex_data["nx"], f_vertex_data["ny"], f_vertex_data["nz"] = f_norms[:, 0], f_norms[:, 1], f_norms[:, 2]
    #f_face_dtype = [("vertex_indices", "i4", (3,))]
    #f_face_data = np.array([(f_faces[i].astype(np.int32),) for i in range(f_faces.shape[0])], dtype=f_face_dtype)
    #PlyData([PlyElement.describe(f_vertex_data, "vertex"), PlyElement.describe(f_face_data, "face")], text=True).write(fast_mesh_path)
    #print(f"Saved fast voxel mesh to {fast_mesh_path} with {f_verts.shape[0]} vertices and {f_faces.shape[0]} faces (iso={fast_iso}, res={fast_res}).")