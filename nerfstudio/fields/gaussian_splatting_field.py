import os
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

from nerfstudio.utils.gaussian_splatting_general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, \
    strip_symmetric, build_scaling_rotation

from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

# Includes geometry opacity, as explained in the gaussial splatting training repository
class GaussianSplattingField(Field):
    def __init__(self, sh_degree: int):
        super().__init__()

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._geometry_opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.appearance_embedding_optimizer = None
        self.appearance_embedding_scheduler = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        geometry_opacities = np.asarray(plydata.elements[0]["geometry_opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._geometry_opacity = nn.Parameter(torch.tensor(geometry_opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_geometry_opacity(self):
        return self.opacity_activation(self._geometry_opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

# Layered version of GaussianSplattingField
class GaussianSplattingFieldLayered(Field):
    def __init__(self, sh_degree: int, num_layers : int):
        super().__init__()
        
        self.num_layers = num_layers
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._geometry_opacity = torch.empty(0) 
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.appearance_embedding_optimizer = None
        self.appearance_embedding_scheduler = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.models = [GaussianSplattingField(sh_degree) for _ in range(num_layers)]
        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)  
    
    # Get model for a given key
    def __getitem__(self, key):
        return self.models[key]
    
    # Get different attributes for given model indices
    ##################################################
    
    def get_scaling(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = range(0, len(self.models))
        scalings = [self.models[m].get_scaling for m in model_idxs]
        return torch.cat((scalings), 0)
    
    def get_rotation(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = range(0, len(self.models))
        rotations = [self.models[m].get_rotation for m in model_idxs]
        return torch.cat((rotations), 0)
    
    def get_xyz(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = range(0, len(self.models))
        xyzs = [self.models[m].get_xyz for m in model_idxs]
        return torch.cat((xyzs), 0)
    
    def get_max_radii2D(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = range(0, len(self.models))
        max_radii2Ds = [self.models[m].max_radii2D for m in model_idxs]
        return torch.cat((max_radii2Ds), 0)

    def get_xyz_gradient_accum(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = range(0, len(self.models))
        xyz_gradient_accums = [self.models[m].xyz_gradient_accum for m in model_idxs]
        return torch.cat((xyz_gradient_accums), 0)
    
    def get_denom(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = range(0, len(self.models))
        denoms = [self.models[m].denom for m in model_idxs]
        return torch.cat((denoms), 0)
    
    def get_features_dc(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = range(0, len(self.models))
        features_dcs = [self.models[m]._features_dc for m in model_idxs]
        return torch.cat((features_dcs), dim=0)
    
    def get_features_rest(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = range(0, len(self.models))
        features_rests = [self.models[m]._features_rest for m in model_idxs]
        return torch.cat((features_rests), dim=0)
    
    def get_features(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = range(0, len(self.models))
        features_dcs = self.get_features_dc(model_idxs)
        features_rests = self.get_features_rest(model_idxs)
        return torch.cat((features_dcs, features_rests), dim=1)
    
    def get_opacity(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = range(0, len(self.models))
        opacities = [self.models[m].get_opacity for m in model_idxs]
        return torch.cat((opacities), dim=0)
    
    def get_geometry_opacity(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = range(0, len(self.models))
        geometry_opacities = [self.models[m].get_geometry_opacity for m in model_idxs]
        return torch.cat((geometry_opacities), dim=0)
    
    def get_spatial_lr_scale(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = range(0, len(self.models))
        spatial_lr_scales = [self.models[m].spatial_lr_scale for m in model_idxs]
        return torch.tensor(spatial_lr_scales)
    
    # Load ply for every layer model
    def load_ply(self, paths, load_iteration):
        for i, sub_path in enumerate(paths):
            path = os.path.join(sub_path, "point_cloud", "iteration_" + str(load_iteration), "point_cloud.ply")
            try:
                self.models[i].load_ply(path)
            except:
                print("Nothing to load for model", i)

        self._xyz = self.models[0]._xyz #self.get_xyz([0])
        self._features_dc = self.models[0]._features_dc #self.get_features_dc([0])
        self._features_rest = self.models[0]._features_rest #self.get_features_rest([0])
        self._opacity = self.models[0]._opacity #self.get_opacity([0])
        self._geometry_opacity = self.models[0]._geometry_opacity #self.get_geometry_opacity([0])
        self._scaling = self.models[0]._scaling #self.get_scaling([0])
        self._rotation = self.models[0]._rotation #self.get_rotation([0])

        self.active_sh_degree = self.max_sh_degree