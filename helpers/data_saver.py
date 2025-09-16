import os
import json
from errno import EEXIST
from os import makedirs, path
import numpy as np
from plyfile import PlyData, PlyElement
from utils.transform_utils import fov_to_focal

def save_ply_to_file(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def save_gaussians_as_ply(path, position, normals, f_dc, f_rest, opacities,
                          scale, rotation, list_of_attributes):
        mkdir_p(os.path.dirname(path))
        dtype_full = [(attribute, 'f4') for attribute in list_of_attributes]
        elements = np.empty(position.shape[0], dtype=dtype_full)
        attributes = np.concatenate((position, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def save_cameras_to_json(path, camlist):
    json_cams = []
    for id, camera in enumerate(camlist):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = camera.R.transpose()
        Rt[:3, 3] = camera.T
        Rt[3, 3] = 1.0
        W2C = np.linalg.inv(Rt)
        pos = W2C[:3, 3]
        rot = W2C[:3, :3]
        serializable_array_2d = [x.tolist() for x in rot]
        camera_entry = {
            'id' : id,
            'img_name' : camera.image_name,
            'width' : camera.width,
            'height' : camera.height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fy' : fov_to_focal(camera.FovY, camera.height),
            'fx' : fov_to_focal(camera.FovX, camera.width)
        }
        json_cams.append(camera_entry)
    with open(path, 'w') as file:
        json.dump(json_cams, file)