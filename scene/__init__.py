import os
import config
import math
import random
import torch
from random import randint
from scene.gaussian import Gaussians
from scene.image import rescale_image_infos
from helpers import read_data_from_path
from utils.loss_utils import l1_loss, ssim
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# from PIL import Image
# import numpy as np

class Scene:
    gaussians : Gaussians
    
    def __init__(self, data_config: config.DataConfigs, opt_config: config.OptimizationConfigs, if_shuffle=True, resolution_scales=[1.0]):
        self.gaussians = Gaussians(opt_config)
        scene_info = read_data_from_path(data_config.source_path, 
                                   data_config.images_dir,
                                   data_config.if_white_background, 
                                   data_config.if_eval)
        self.scene_extent = scene_info.nerf_normalization["radius"]
        self.viewpoint_stack = None
        
        self.train_images = {}
        self.test_images = {}
        if if_shuffle:
            random.shuffle(scene_info.train_images)
            random.shuffle(scene_info.test_images) 
        for resolution_scale in resolution_scales:
            self.train_images[resolution_scale] = rescale_image_infos(scene_info.train_images, 
                                                                       resolution_scale, 
                                                                       data_config.resolution_scale_factor, 
                                                                       data_config.data_device)
            self.test_images[resolution_scale] = rescale_image_infos(scene_info.test_images, 
                                                                       resolution_scale, 
                                                                       data_config.resolution_scale_factor, 
                                                                       data_config.data_device)
        self.gaussians.init_gaussians_with_pc(scene_info.point_cloud, self.scene_extent)

        bg_color = [1, 1, 1] if data_config.if_white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def training_setup(self, opt_config):
        self.gaussians.training_setup(opt_config)

    def optimization(self, iteration, opt_config, scale=1.0):
        self.gaussians.update_learing_rate(iteration)
        if iteration % 1000 == 0:
            self.gaussians.level_up_sh_degree()
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.train_images[scale].copy()
        self.viewpoint_camera = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
        
        render_pkg = self.render()
        self.render_image, self.render_viewspace_point_tensor, self.render_visibility_filter, self.render_radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = self.viewpoint_camera.original_image.cuda()
        Ll1 = l1_loss(self.render_image, gt_image)
        loss = (1.0 - opt_config.lambda_dssim) * Ll1 + opt_config.lambda_dssim * (1.0 - ssim(self.render_image, gt_image))
        loss.backward()

        # if iteration == 1:
        #      with open('opacoxel/test_output/opacoxel_viewpoint.txt', 'w', encoding='utf-8') as viewpoint_f:
        #           pass
        # with open('opacoxel/test_output/opacoxel_viewpoint.txt', 'a', encoding='utf-8') as viewpoint_f:
        #     viewpoint_f.write(str(iteration) + ": " + self.viewpoint_camera.image_name + "\n")
        #     for row in self.viewpoint_camera.R_transpose:
        #         line = ','.join(f'{x:.4f}' for x in row)
        #         viewpoint_f.write(line)
        #         viewpoint_f.write('\n')
        #     T_py = self.viewpoint_camera.T.tolist()
        #     T_save = str(T_py) 
        #     viewpoint_f.write(T_save) 
        #     viewpoint_f.write('\n')
        #     viewpoint_f.write('-----------------------\n')

        # if (iteration - 1) % 1000 == 0:
        #     image_np = self.render_image.detach().cpu().numpy()
        #     image_np = np.transpose(image_np, (1, 2, 0))
        #     array = np.array(image_np*255.0, dtype=np.byte)  
        #     image = Image.fromarray(array, "RGB")  
        #     image.save("opacoxel/test_output/output_" + str(iteration) + ".png" )

    def get_train_images(self, scale=1.0):
        return self.train_images[scale]
    
    def get_test_images(self, scale=1.0):
        return self.test_images[scale] 
    
    def render(self, scaling_modifier = 1.0, debug=False):
        screenspace_points = torch.zeros_like(self.gaussians._position, 
                                              dtype=self.gaussians.get_position.dtype, 
                                              requires_grad=True, 
                                              device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        tanfovx = math.tan(self.viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(self.viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.viewpoint_camera.image_height),
            image_width=int(self.viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.background,
            scale_modifier=scaling_modifier,
            viewmatrix=self.viewpoint_camera.world_view_transform_transpose,
            projmatrix=self.viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians._active_sh_degree,
            campos=self.viewpoint_camera.camera_center,
            prefiltered=False,
            debug=debug)

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points
        means3D = self.gaussians.get_position
        scales = self.gaussians.get_scaling
        rotations = self.gaussians.get_rotation
        shs = self.gaussians.get_features
        opacity = self.gaussians.get_opacity

        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None)
    
        return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
    
    def gaussian_densify(self, iteration, data_config: config.DataConfigs, opt_config: config.OptimizationConfigs):
        if iteration < opt_config.densify_until_iter:
            self.gaussians.get_max_radii2D[self.render_visibility_filter] = torch.max(self.gaussians.get_max_radii2D[self.render_visibility_filter], self.render_radii[self.render_visibility_filter])
            self.gaussians.update_densification_stats(self.render_viewspace_point_tensor, self.render_visibility_filter)
            if iteration > opt_config.densify_from_iter and iteration % opt_config.densification_interval == 0:
                size_threshold = 20 if iteration > opt_config.opacity_reset_interval else None
                self.gaussians.densify_and_prune(opt_config.densify_grad_threshold, 0.005, self.scene_extent, size_threshold, iteration)
            if iteration % opt_config.opacity_reset_interval == 0 or (data_config.if_white_background and iteration == opt_config.densify_from_iter):
                self.gaussians.reset_opacity()
    
    def save_scene_as_ply(self, model_path, iteration):
        ply_path = os.path.join(model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_as_ply(os.path.join(ply_path, "point_cloud.ply"))
