import config
import torch
from torch import nn
import numpy as np
from typing import NamedTuple
from simple_knn._C import distCUDA2
from utils.sh_utils import RGB_to_sh
from utils.transform_utils import build_covariance_from_scaling_rotation, quat_to_rot_mat_torch
from helpers import BasicPointCloud
from helpers import save_gaussians_as_ply

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class Gaussians:
    def __init__(self, opt_config: config.OptimizationConfigs):
        self._position = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._opacity = torch.empty(0)
        self._opacity_field = None  # Optional external field providing alpha

        self._active_sh_degree = 0
        self._max_sh_degree = opt_config.sh_degree
        self._percent_dense = opt_config.percent_dense
        self._spatial_lr_scale = 0

        self._max_radii2D = torch.empty(0)
        self._position_gradient_accum = torch.empty(0)
        self._position_gradient_count = torch.empty(0)

        self._scaling_activation = torch.exp
        self._scaling_inverse_activation = torch.log
        self._rotation_activation = torch.nn.functional.normalize
        self._opacity_activation = torch.sigmoid
        self._inverse_opacity_actvation = inverse_sigmoid
        self._covariance_activation = build_covariance_from_scaling_rotation

        self._optimizer = None

    @property
    def get_position(self):
        return self._position

    @property
    def get_scaling(self):
        return self._scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self._rotation_activation(self._rotation)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        if self._opacity_field is not None and isinstance(self._position, torch.Tensor) and self._position.numel() > 0:
            # Sample from field using current 3D means; returns [N,1]
            try:
                return self._opacity_field.sample_alpha(self._position)
            except Exception:
                # Fallback to internal opacity if sampling fails
                return self._opacity_activation(self._opacity)
        return self._opacity_activation(self._opacity)

    def set_opacity_field(self, field):
        """Attach external opacity field (Opacoxel)."""
        self._opacity_field = field
    
    @property
    def get_sh_degree(self):
        return self._active_sh_degree

    @property
    def get_max_radii2D(self):
        return self._max_radii2D

    def get_covariance(self, scaling_modifier = 1):
        return self._covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def level_up_sh_degree(self):
        if self._active_sh_degree < self._max_sh_degree:
            self._active_sh_degree += 1

    def init_gaussians_with_pc(self, pc: BasicPointCloud,  spatial_lr_scale : float):
        self._spatial_lr_scale = spatial_lr_scale
        point_cloud = torch.tensor(np.asarray(pc.points)).float().cuda()
        color = RGB_to_sh(torch.tensor(np.asarray(pc.colors)).float().cuda())
        features = torch.zeros((color.shape[0], 3, (self._max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = color
        features[:, :3, 1:] = 0.0
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pc.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        print("Number of points at initialisation : ", point_cloud.shape[0])

        self._position = nn.Parameter(point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._max_radii2D = torch.zeros((self.get_position.shape[0]), device="cuda")
        self._position_gradient_accum = torch.zeros((self.get_position.shape[0], 1), device="cuda")
        self._position_gradient_count = torch.zeros((self.get_position.shape[0], 1), device="cuda")

    def init_gaussians_with_plyfile(self, ply_data):
        xyz = np.stack((np.asarray(ply_data.elements[0]["x"]),
                        np.asarray(ply_data.elements[0]["y"]),
                        np.asarray(ply_data.elements[0]["z"])),  axis=1)
        opacities = np.asarray(ply_data.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(ply_data.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(ply_data.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(ply_data.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(ply_data.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(ply_data.elements[0][attr_name])
        
        rot_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def training_setup(self, opt_config: config.OptimizationConfigs):
        l = [
            {'params': [self._position], 'lr': opt_config.position_lr_init * self._spatial_lr_scale, "name": "position"},
            {'params': [self._features_dc], 'lr': opt_config.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': opt_config.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opt_config.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': opt_config.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': opt_config.rotation_lr, "name": "rotation"}
        ]
        self._position_scheduler_args = get_expon_lr_func(lr_init=opt_config.position_lr_init*self._spatial_lr_scale,
                                                    lr_final=opt_config.position_lr_final*self._spatial_lr_scale,
                                                    lr_delay_mult=opt_config.position_lr_delay_mult,
                                                    max_steps=opt_config.position_lr_max_steps)
        self._optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    def update_learing_rate(self, iteration):
        for param_group in self._optimizer.param_groups:
            if param_group["name"] == "position":
                lr = self._position_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr 

    def update_densification_stats(self, viewspace_point_tensor, update_filter):
        self._position_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self._position_gradient_count[update_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration):
        grads = self._position_gradient_accum / self._position_gradient_count
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
             big_points_vs = self._max_radii2D > max_screen_size
             big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
             prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
    
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_position.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self._percent_dense*scene_extent)
        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quat_to_rot_mat_torch(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_position = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_position[selected_pts_mask].repeat(N, 1)
        new_scaling = self._scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_position, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self._percent_dense*scene_extent)
        
        new_position = self._position[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_position, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densification_postfix(self, new_position, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"position": new_position,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._position = optimizable_tensors["position"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._position_gradient_accum = torch.zeros((self.get_position.shape[0], 1), device="cuda")
        self._position_gradient_count = torch.zeros((self.get_position.shape[0], 1), device="cuda")
        self._max_radii2D = torch.zeros((self.get_position.shape[0]), device="cuda")
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self._optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self._optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self._optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self._optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._position = optimizable_tensors["position"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
    
        self._position_gradient_accum = self._position_gradient_accum[valid_points_mask]
        self._position_gradient_count = self._position_gradient_count[valid_points_mask]
        self._max_radii2D = self._max_radii2D[valid_points_mask]
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self._optimizer.param_groups:
            stored_state = self._optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self._optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self._optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self._optimizer.param_groups:
            if group["name"] == name:
                stored_state = self._optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self._optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self._optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_as_ply(self, path):
        position = self._position.detach().cpu().numpy()
        normals = np.zeros_like(position)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        save_gaussians_as_ply(path,
                              position,
                              normals,
                              f_dc,
                              f_rest,
                              opacities,
                              scale,
                              rotation,
                              self.construct_list_of_attributes())


