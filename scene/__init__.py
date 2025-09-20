import os
import config
import math
import random
import torch
from random import randint
from scene.gaussian import Gaussians
from scene.image import rescale_image_infos
from scene.opacoxel import Opacoxel
from helpers import read_data_from_path, read_ply_data, save_cameras_to_json
from utils.loss_utils import l1_loss, ssim
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class Scene:
    gaussians : Gaussians

    def __init__(self, data_config: config.DataConfigs, opt_config: config.OptimizationConfigs, load_scene_iter=None, if_shuffle=True, resolution_scales=[1.0]):
        self.model_path = data_config.model_path
        self.load_scene_iter = None
        self.gaussians = Gaussians(opt_config)
        self.train_images = {}
        self.test_images = {}
        self.viewpoint_stack = None
        bg_color = [1, 1, 1] if data_config.if_white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if load_scene_iter is not None:
            if load_scene_iter == -1:
                self.load_scene_iter = max([int(fname.split("_")[-1]) for fname in os.listdir(os.path.join(self.model_path, "point_cloud"))])
            else:
                self.load_scene_iter = load_scene_iter

        scene_info = read_data_from_path(data_config.source_path, 
                                   data_config.images_dir,
                                   data_config.if_white_background, 
                                   data_config.if_eval)
        if not self.load_scene_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            camlist = []
            if scene_info.test_images:
                camlist.extend(scene_info.test_images)
            if scene_info.train_images:
                camlist.extend(scene_info.train_images)
            save_cameras_to_json(os.path.join(self.model_path, "cameras.json"), camlist)
    
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
        self.scene_extent = scene_info.nerf_normalization["radius"]
            
        if self.load_scene_iter:
            ply_data = read_ply_data(os.path.join(self.model_path,
                                                 "point_cloud",
                                                "iteration_" + str(self.load_scene_iter),
                                                "point_cloud.ply"))
            self.gaussians.init_gaussians_with_plyfile(ply_data)
        else:
            self.gaussians.init_gaussians_with_pc(scene_info.point_cloud, self.scene_extent)

        # Initialize Opacoxel field bounds from point cloud if available
        # Determine scene AABB from initial positions
        with torch.no_grad():
            if isinstance(self.gaussians.get_position, torch.Tensor) and self.gaussians.get_position.numel() > 0:
                pts = self.gaussians.get_position.detach().cpu()
                mins = pts.min(dim=0).values
                maxs = pts.max(dim=0).values
                # Pad a bit to be safe
                pad = 0.05 * (maxs - mins).max().item() if torch.isfinite((maxs - mins).max()) else 0.1
                bounds = (float(mins[0] - pad), float(maxs[0] + pad),
                          float(mins[1] - pad), float(maxs[1] + pad),
                          float(mins[2] - pad), float(maxs[2] + pad))
            else:
                # Fallback cube around origin
                bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

        # Create field; resolution can be tuned
        self.opacity_field = Opacoxel(bounds=bounds, resolution=(64, 64, 64), device="cuda")
        # Initialize field from SfM points (use gaussians init positions)
        with torch.no_grad():
            if isinstance(self.gaussians.get_position, torch.Tensor) and self.gaussians.get_position.numel() > 0:
                self.opacity_field.initialize_from_point_cloud(self.gaussians.get_position.detach())
        # Attach to gaussians, so rasterizer will use sampled alpha
        self.gaussians.set_opacity_field(self.opacity_field)

    def training_setup(self, opt_config):
        self.gaussians.training_setup(opt_config)
        # Add an optimizer for the Opacoxel logits
        # Keep separate to manage schedulers independently
        self.opacity_optimizer = torch.optim.Adam([
            {"params": [self.opacity_field.logit_grid], "lr": opt_config.opacity_lr}
        ], lr=0.0)

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
        # Opacity sampled from attached field inside Gaussians.get_opacity
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
