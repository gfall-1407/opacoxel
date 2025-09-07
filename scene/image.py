import torch
from torch import nn
import numpy as np
from utils.transform_utils import build_world_to_view_mat_withSceneScale, build_projection_matrix

WARNED = False

class Image(nn.Module):
    def __init__(self, dataset_id, R, R_transpose, T, FoVx, FoVy, image, alpha_mask, image_name, cam_uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"):
        super(Image, self).__init__()
        self.dataset_id = dataset_id
        self.R = R
        self.R_transpose = R_transpose
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.cam_uid = cam_uid
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if alpha_mask is not None:
            self.original_image *= alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform_transpose = torch.tensor(build_world_to_view_mat_withSceneScale(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix_transpose = build_projection_matrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform_transpose.unsqueeze(0).bmm(self.projection_matrix_transpose.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform_transpose.inverse()[3, :3]

def rescale_image_infos(image_infos, resolution_scale, resolution_scale_fatcor, data_device):
    image_list = []
    factor = resolution_scale_fatcor
    for id, img_info in enumerate(image_infos):
        orig_w, orig_h = img_info.image.size
        if factor in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * factor)), round(orig_h/(resolution_scale * factor))
        else:
            if factor == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / factor
        
            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))
        
        resized_image_rgb = pil_to_torch(img_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        alpha_mask = None
        if resized_image_rgb.shape[1] == 4:
            alpha_mask = resized_image_rgb[3:4, ...]
        gt_image = resized_image_rgb[:3, ...]
        alpha_mask = None
        if resized_image_rgb.shape[1] == 4:
            alpha_mask = resized_image_rgb[3:4, ...]
        image_list.append(Image(dataset_id=id, R=img_info.R, 
                                R_transpose=img_info.R_transpose, T=img_info.T,
                                FoVx=img_info.FovX, FoVy=img_info.FovY,
                                image=gt_image, alpha_mask=alpha_mask,
                                image_name=img_info.image_name, cam_uid=img_info.uid,
                                data_device=data_device))
    return image_list

def pil_to_torch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)