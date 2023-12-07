import os
import json
from matplotlib import cm
import numpy as np
import torch
import math
import torchvision
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.fields.gaussian_splatting_field import GaussianSplattingField, GaussianSplattingFieldLayered
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from nerfstudio.utils.gaussian_splatting_sh_utils import eval_sh, RGB2SH

from nerfstudio.cameras.gaussian_splatting_camera import Camera as GaussianSplattingCamera
from nerfstudio.utils.gaussian_splatting_graphics_utils import getWorld2View2, focal2fov, fov2focal
from torchvision.transforms.functional import rgb_to_grayscale
from nerfstudio.utils import constants

@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    _target: Type = field(
        default_factory=lambda: GaussianSplatting
    )

    background_color: str = "black"

    sh_degree: int = 3
    
@dataclass
class GaussianSplattingModelConfigLayered(ModelConfig):
    _target: Type = field(
        default_factory=lambda: GaussianSplattingLayered
    )

    background_color: str = "black"

    sh_degree: int = 3


class PipelineParams():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


class GaussianSplatting(Model):
    config: GaussianSplattingModelConfig
    model_path: str
    load_iteration: int
    ref_orientation: str
    orientation_transform: torch.Tensor
    gaussian_model: GaussianSplattingField

    def __init__(
            self,
            config: ModelConfig,
            scene_box: SceneBox,
            num_train_data: int,
            model_path: str = None,
            load_iteration: int = -1,
            orientation_transform: torch.Tensor = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.load_iteration = load_iteration
        self.orientation_transform = orientation_transform
        self.pipeline_params = PipelineParams()
        if self.config.background_color == "black":
            self.bg_color = [0, 0, 0]
        else:
            self.bg_color = [1, 1, 1]

        super().__init__(config, scene_box, num_train_data)

    def populate_modules(self):
        super().populate_modules()

        # get iteration
        if self.load_iteration == -1:
            self.load_iteration = self.search_for_max_iteration(os.path.join(self.model_path, "point_cloud"))
        print("Loading trained model at iteration {}".format(self.load_iteration))

        # load gaussian model
        self.gaussian_model = GaussianSplattingField(sh_degree=self.config.sh_degree)

        self.gaussian_model.load_ply(os.path.join(self.model_path,
                                                  "point_cloud",
                                                  "iteration_" + str(self.load_iteration),
                                                  "point_cloud.ply"))

    @staticmethod
    def search_for_max_iteration(folder):
        saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
        return max(saved_iters)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        viewpoint_camera = self.ns2gs_camera(camera_ray_bundle.camera)

        background = torch.tensor(self.bg_color, dtype=torch.float32, device=camera_ray_bundle.origins.device)

        render_results = self.render(
            viewpoint_camera=viewpoint_camera,
            pc=self.gaussian_model,
            pipe=self.pipeline_params,
            bg_color=background
        )

        render = render_results["render"]
        rgb = torch.permute(torch.clamp(render, max=1.), (1, 2, 0))

        return {
            "rgb": rgb,
        }

    def ns2gs_camera(self, ns_camera):
        c2w = torch.clone(ns_camera.camera_to_worlds)
        c2w = torch.concatenate([c2w, torch.tensor([[0, 0, 0, 1]], device=ns_camera.camera_to_worlds.device)], dim=0)

        # reorient
        if self.orientation_transform is not None:
            c2w = torch.matmul(self.orientation_transform.to(c2w.device), c2w)

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w.cpu().numpy())
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        FovY = focal2fov(ns_camera.fy, ns_camera.height)
        FovX = focal2fov(ns_camera.fx, ns_camera.width)

        return GaussianSplattingCamera(
            R=R,
            T=T,
            width=ns_camera.width,
            height=ns_camera.height,
            FoVx=FovX,
            FoVy=FovY,
            data_device=ns_camera.camera_to_worlds.device,
        )

    @staticmethod
    def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        # bg_color = torch.rand_like(bg_color)
        
        flipped_xyz = pc.get_xyz
        flipped_xyz[:, [1, 2]] = flipped_xyz[:, [2, 1]]
        flipped_xyz[..., 1] = -flipped_xyz[..., 1]
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(flipped_xyz, dtype=flipped_xyz.dtype, requires_grad=True,
                                              device=bg_color.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = flipped_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity
        geometry_opacity = pc.get_geometry_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (flipped_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color
        
        if constants.GEOM_FLAG:
            # mask_shs = shs.clone().detach()
            # ones_rgb = torch.ones_like(mask_shs[:, :3, 0 ])
            # mask_shs[:, :3, 0 ] = RGB2SH(ones_rgb)
            # mask_shs[:, 3:, 1:] = 0.0
            cmap = cm.get_cmap("plasma")
            campos = viewpoint_camera.camera_center
            depths = -torch.norm(means3D - campos, dim=1)
            depths -= torch.min(depths)
            depths /= torch.max(depths)
            mask_shs = torch.zeros_like(shs)
            mask_shs[:, 0] = torch.from_numpy(2.5 * cmap(depths.cpu().numpy())[:, :3]).to(mask_shs.device)
            # mask_shs[:, 0, 0] = 2.5 * (1.0 - depths)
            # depth_opacity = torch.ones_like(geometry_opacity)
        
            # Rasterize visible Gaussians to image, obtain their radii (on screen). 
            rendered_image, radii_mask = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = mask_shs,
                colors_precomp = None,
                opacities = geometry_opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
               
            # # rendered_mask = rgb_to_grayscale(rendered_mask)
            # return {"render": rendered_image,
            #     "mask": rendered_mask,
            #     "viewspace_points": screenspace_points,
            #     "visibility_filter": radii > 0,
            #     "radii": radii}
        else:
            # Rasterize visible Gaussians to image, obtain their radii (on screen).
            rendered_image, radii = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii}
        
class GaussianSplattingLayered(Model):
    config: GaussianSplattingModelConfigLayered
    model_path: str
    load_iteration: int
    ref_orientation: str
    orientation_transform: torch.Tensor
    gaussian_model: GaussianSplattingFieldLayered

    def __init__(
            self,
            config: ModelConfig,
            scene_box: SceneBox,
            num_train_data: int,
            model_path: str = None,
            load_iteration: int = -1,
            orientation_transform: torch.Tensor = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.sub_paths = [os.path.join(model_path, p) for p in  os.listdir(model_path)]
        self.sub_paths = list(filter(lambda p: os.path.isdir(p), self.sub_paths))  
        self.sub_paths.sort()
        self.num_layers = len(self.sub_paths)
        self.load_iteration = load_iteration
        self.orientation_transform = orientation_transform
        self.pipeline_params = PipelineParams()
        if self.config.background_color == "black":
            self.bg_color = [0, 0, 0]
        else:
            self.bg_color = [1, 1, 1]

        super().__init__(config, scene_box, num_train_data)

    def populate_modules(self):
        super().populate_modules()

        # get iteration
        if self.load_iteration == -1:
            self.load_iteration = self.search_for_max_iteration(os.path.join(self.sub_paths[0], "point_cloud"))
        print("Loading trained model at iteration {}".format(self.load_iteration))

        # load gaussian model
        self.gaussian_model = GaussianSplattingFieldLayered(sh_degree=self.config.sh_degree, num_layers=self.num_layers)

        self.gaussian_model.load_ply(self.sub_paths, self.load_iteration)
        #os.path.join(self.model_path,
            # "point_cloud",
            # "iteration_" + str(self.load_iteration),
            # "point_cloud.ply")

    @staticmethod
    def search_for_max_iteration(folder):
        saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
        return max(saved_iters)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        viewpoint_camera = self.ns2gs_camera(camera_ray_bundle.camera)

        background = torch.tensor(self.bg_color, dtype=torch.float32, device=camera_ray_bundle.origins.device)

        render_results = self.render(
            viewpoint_camera=viewpoint_camera,
            pc=self.gaussian_model,
            pipe=self.pipeline_params,
            bg_color=background
        )

        render = render_results["render"]
        rgb = torch.permute(torch.clamp(render, max=1.), (1, 2, 0))

        return {
            "rgb": rgb,
        }

    def ns2gs_camera(self, ns_camera):
        c2w = torch.clone(ns_camera.camera_to_worlds)
        c2w = torch.concatenate([c2w, torch.tensor([[0, 0, 0, 1]], device=ns_camera.camera_to_worlds.device)], dim=0)

        # reorient
        if self.orientation_transform is not None:
            c2w = torch.matmul(self.orientation_transform.to(c2w.device), c2w)

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w.cpu().numpy())
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        FovY = focal2fov(ns_camera.fy, ns_camera.height)
        FovX = focal2fov(ns_camera.fx, ns_camera.width)

        return GaussianSplattingCamera(
            R=R,
            T=T,
            width=ns_camera.width,
            height=ns_camera.height,
            FoVx=FovX,
            FoVy=FovY,
            data_device=ns_camera.camera_to_worlds.device,
        )

    @staticmethod
    def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """      
        # bg_color = torch.rand_like(bg_color)
        
        
        # correct layer range
        if constants.LAYER_RANGE[0] >= pc.num_layers:
            constants.LAYER_RANGE[0] = pc.num_layers - 1
        if constants.LAYER_RANGE[1] >= pc.num_layers:
            constants.LAYER_RANGE[1] = constants.LAYER_RANGE[0]
            
        if constants.LAYER_RANGE[0] < 0:
            constants.LAYER_RANGE[0] = 0
        if constants.LAYER_RANGE[1] < 0:
            constants.LAYER_RANGE[1] = constants.LAYER_RANGE[0]
        model_idxs = range(constants.LAYER_RANGE[0], constants.LAYER_RANGE[1] + 1)
        
        flipped_xyz_all = pc.get_xyz(model_idxs)
        flipped_xyz = flipped_xyz_all[~(torch.isnan(flipped_xyz_all).any(dim=1))]

        flipped_xyz[:, [1, 2]] = flipped_xyz[:, [2, 1]]
        flipped_xyz[..., 1] = -flipped_xyz[..., 1]
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(flipped_xyz, dtype=flipped_xyz.dtype, requires_grad=True,
                                              device=bg_color.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = flipped_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity(model_idxs)
        opacity = opacity[~(torch.isnan(flipped_xyz_all).any(dim=1))]
        # if constants.GEOM_FLAG:
        geometry_opacity = pc.get_geometry_opacity(model_idxs)
        geometry_opacity = geometry_opacity[~(torch.isnan(flipped_xyz_all).any(dim=1))]

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier, model_idxs)
        else:
            scales = pc.get_scaling(model_idxs)[~(torch.isnan(flipped_xyz_all).any(dim=1))]
            rotations = pc.get_rotation(model_idxs)[~(torch.isnan(flipped_xyz_all).any(dim=1))]

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features(model_idxs)[~(torch.isnan(flipped_xyz_all).any(dim=1))].transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (flipped_xyz- viewpoint_camera.camera_center.repeat(pc.get_features(model_idxs).shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features(model_idxs)[~(torch.isnan(flipped_xyz_all).any(dim=1))]
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=geometry_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        
        if constants.GEOM_FLAG:
            # mask_shs = shs.clone().detach()
            # ones_rgb = torch.ones_like(mask_shs[:, :3, 0 ])
            # mask_shs[:, :3, 0 ] = RGB2SH(ones_rgb)
            # mask_shs[:, 3:, 1:] = 0.0
            cmap = cm.get_cmap("plasma")
            campos = viewpoint_camera.camera_center
            depths = -torch.norm(means3D - campos, dim=1)
            depths -= torch.min(depths)
            depths /= torch.max(depths)
            depth_shs = torch.zeros_like(shs)
            depth_shs[:, 0] = torch.from_numpy(2.5 * cmap(depths.cpu().numpy())[:, :3]).to(depth_shs.device)

            # mask_shs[:, 0, 0] = 2.5 * (1.0 - depths)
            # depth_opacity = torch.ones_like(geometry_opacity)
        
            # Rasterize visible Gaussians to image, obtain their radii (on screen). 
            rendered_image, radii_mask = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = depth_shs,
                colors_precomp = None,
                opacities = geometry_opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
            
            # rendered_image = rgb_to_grayscale(rendered_image)
            # rendered_image = torch.clip(rendered_image, 0, 1)
               
            # # rendered_mask = rgb_to_grayscale(rendered_mask)
            # return {"render": rendered_image,
            #     "mask": rendered_mask,
            #     "viewspace_points": screenspace_points,
            #     "visibility_filter": radii > 0,
            #     "radii": radii}

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii}
