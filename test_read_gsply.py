import numpy as np
from typing import NamedTuple
from plyfile import PlyData
from scene import Opacoxels

class GaussianPLY(NamedTuple):
    positions : np.array
    scales : np.array
    rotations : np.array 
    features_dc : np.array
    features_rest : np.array
    opacities : np.array

def read_gd_ply(path, max_sh_degree):
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
    gaussians = read_gd_ply("PLYs/point_cloud_1000.ply", 3)
    bbox = compute_bbox(gaussians.positions)
    opacoxels = Opacoxels()