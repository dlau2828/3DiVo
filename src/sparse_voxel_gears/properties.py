# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

from src.utils import octree_utils
from src.utils.fuser_utils import rgb_fusion
from src.utils.activation_utils import rgb2shzero

import svraster_cuda


class SVProperties:

    @property
    def num_voxels(self):
        return len(self.octpath)

    @property
    def num_grid_pts(self):
        return len(self.grid_pts_key)

    @property
    def scene_min(self):
        return self.scene_center - 0.5 * self.scene_extent

    @property
    def scene_max(self):
        return self.scene_center + 0.5 * self.scene_extent

    @property
    def inside_min(self):
        return self.scene_center - 0.5 * self.inside_extent

    @property
    def inside_max(self):
        return self.scene_center + 0.5 * self.inside_extent

    @property
    def outside_level(self):
        return (self.scene_extent / self.inside_extent).log2().round().long().item()

    @property
    def bounding(self):
        return torch.stack([self.scene_min, self.scene_max])

    @property
    def inside_mask(self):
        isin = ((self.inside_min < self.vox_center) & (self.vox_center < self.inside_max)).all(1)
        return isin

    @property
    def sh0(self):
        return self._sh0

    @property
    def shs(self):
        return self._shs

    @property
    def subdivision_priority(self):
        return self._subdiv_p.grad

    def reset_subdivision_priority(self):
        self._subdiv_p.grad = None

    @property
    def signature(self):
        # Signature to check if the voxel grid layout is updated
        return (self.num_voxels, id(self.octpath), id(self.octlevel))

    def _check_derived_voxel_attr(self):
        # Lazy computation of inverse voxel sizes
        signature = self.signature
        need_recompute = not hasattr(self, '_check_derived_voxel_attr_signature') or \
                         self._check_derived_voxel_attr_signature != signature
        if need_recompute:
            self._vox_center, self._vox_size = octree_utils.octpath_decoding(
                self.octpath, self.octlevel, self.scene_center, self.scene_extent)
            self._grid_pts_key, self._vox_key = octree_utils.build_grid_pts_link(self.octpath, self.octlevel)
            self._check_derived_voxel_attr_signature = signature

    @property
    def vox_center(self):
        self._check_derived_voxel_attr()
        return self._vox_center

    @property
    def vox_size(self):
        self._check_derived_voxel_attr()
        return self._vox_size

    @property
    def grid_pts_key(self):
        self._check_derived_voxel_attr()
        return self._grid_pts_key

    @property
    def vox_key(self):
        self._check_derived_voxel_attr()
        return self._vox_key

    @property
    def vox_size_inv(self):
        # Lazy computation of inverse voxel sizes
        signature = self.signature
        need_recompute = not hasattr(self, '_vox_size_inv_signature') or \
                         self._vox_size_inv_signature != signature
        if need_recompute:
            self._vox_size_inv = 1 / self.vox_size
            self._vox_size_inv_signature = signature
        return self._vox_size_inv

    @property
    def grid_pts_xyz(self):
        # Lazy computation of grid points xyz
        signature = self.signature
        need_recompute = not hasattr(self, '_grid_pts_xyz_signature') or \
                         self._grid_pts_xyz_signature != signature
        if need_recompute:
            self._grid_pts_xyz = octree_utils.compute_gridpoints_xyz(
                self.grid_pts_key, self.scene_center, self.scene_extent)
            self._grid_pts_xyz_signature = signature
        return self._grid_pts_xyz

    @torch.no_grad()
    def reset_sh_from_cameras(self, cameras):
        self._sh0.data.copy_(rgb2shzero(rgb_fusion(self, cameras)))
        self._shs.data.zero_()

    def apply_tv_on_density_field(self, lambda_tv_density):
        if self._geo_grid_pts.grad is None:
            self._geo_grid_pts.grad = torch.zeros_like(self._geo_grid_pts.data)
        svraster_cuda.grid_loss_bw.total_variation(
            grid_pts=self._geo_grid_pts,
            vox_key=self.vox_key,
            weight=lambda_tv_density,
            vox_size_inv=self.vox_size_inv,
            no_tv_s=True,
            tv_sparse=False,
            grid_pts_grad=self._geo_grid_pts.grad)
