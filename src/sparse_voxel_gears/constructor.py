# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import numpy as np
import svraster_cuda

from src.utils.activation_utils import rgb2shzero
from src.utils import octree_utils

class SVConstructor:

    def model_init(self,
                   bounding,           # Scene bound [min_xyz, max_xyz]
                   outside_level,      # Number of Octree levels for background
                   init_n_level=6,     # Starting from (2^init_n_level)^3 voxels
                   init_out_ratio=2.0, # Number of voxel ratio for outside (background region)
                   sh_degree_init=3,   # Initial activated sh degree
                   geo_init=-10.0,     # Init pre-activation density
                   sh0_init=0.5,       # Init voxel colors in range [0,1]
                   shs_init=0.0,       # Init coefficients of higher-degree sh
                   cameras=None,       # Cameras that helps voxel allocation
                   ):

        assert outside_level <= svraster_cuda.meta.MAX_NUM_LEVELS

        # Define scene bound
        center = (bounding[0] + bounding[1]) * 0.5
        extent = max(bounding[1] - bounding[0])
        self.scene_center, self.scene_extent, self.inside_extent = get_scene_bound_tensor(
            center=center, extent=extent, outside_level=outside_level)

        # Init voxel layout.
        # The world is seperated into inside (main foreground) and outside (background) regions.
        in_path, in_level = octlayout_inside_uniform(
            scene_center=self.scene_center,
            scene_extent=self.scene_extent,
            outside_level=outside_level,
            n_level=init_n_level,
            cameras=cameras,
            filter_zero_visiblity=(cameras is not None),
            filter_near=-1)

        if outside_level == 0:
            # Object centric bounded scenes
            ou_path = torch.empty([0, 1], dtype=in_path.dtype, device="cuda")
            ou_level = torch.empty([0, 1], dtype=in_level.dtype, device="cuda")
        else:
            min_num = len(in_path) * init_out_ratio
            max_level = outside_level + init_n_level
            ou_path, ou_level = octlayout_outside_heuristic(
                scene_center=self.scene_center,
                scene_extent=self.scene_extent,
                outside_level=outside_level,
                cameras=cameras,
                min_num=min_num,
                max_level=max_level,
                filter_near=-1)

        self.octpath = torch.cat([ou_path, in_path])
        self.octlevel = torch.cat([ou_level, in_level])

        self.active_sh_degree = min(sh_degree_init, self.max_sh_degree)

        # Init trainable parameters
        self._geo_grid_pts = torch.full(
            [self.num_grid_pts, 1], geo_init,
            dtype=torch.float32, device="cuda").requires_grad_()

        self._sh0 = torch.full(
            [self.num_voxels, 3], rgb2shzero(sh0_init),
            dtype=torch.float32, device="cuda").requires_grad_()

        self._shs = torch.full(
            [self.num_voxels, (self.max_sh_degree+1)**2 - 1, 3], shs_init,
            dtype=torch.float32, device="cuda").requires_grad_()

        # Subdivision priority trackor
        self._subdiv_p = torch.ones(
            [self.num_voxels, 1],
            dtype=torch.float32, device="cuda").requires_grad_()

    def octpath_init(self,
                  scene_center,
                  scene_extent,
                  octpath,       # Nx1 octpath.
                  octlevel,      # Nx1 or scalar for the Octree level of each voxel.

                  # The following are model parameters.
                  # If the input are tensors, the gradient of rendering can be backprop to them.
                  # Otherwise, it creates new trainable tensors.
                  rgb=0.5,       # Nx3 or scalar for voxel color in range of 0~1.
                  shs=0.0,       # NxDx3 or scalar for voxel higher-deg sh coefficient.
                  density=-10.,  # Nx8 or Ngridx1 or scalar for voxel density field.
                                 # The order is [0,0,0] => [0,0,1] => [0,1,0] => [0,1,1] ...
                  reduce_density=False,  # Whether to merge grid points if density is Nx8.
                  ):

        self.scene_center, self.scene_extent, self.inside_extent = get_scene_bound_tensor(
            center=scene_center, extent=scene_extent)

        assert torch.is_tensor(octpath)
        octlevel = get_octlevel_tensor(octlevel, num_voxels=len(octpath))

        self.octpath = octpath.view(-1, 1).contiguous()
        self.octlevel = octlevel.view(-1, 1).contiguous()
        assert len(self.octpath) == len(self.octlevel)

        # Subdivision priority trackor
        self._subdiv_p = torch.ones(
            [self.num_voxels, 1],
            dtype=torch.float32, device="cuda").requires_grad_()

        # Setup appearence parameters
        if torch.is_tensor(rgb):
            assert rgb.shape == (self.num_voxels, 3)
            self._sh0 = rgb2shzero(rgb.contiguous().cuda())
        else:
            self._sh0 = torch.full(
                [self.num_voxels, 3], rgb2shzero(rgb),
                dtype=torch.float32, device="cuda").requires_grad_()

        if torch.is_tensor(shs):
            assert shs.shape == (self.num_voxels, (self.max_sh_degree+1)**2 - 1, 3)
            self.shs = shs.contiguous().cuda()
        else:
            self._shs = torch.full(
                [self.num_voxels, (self.max_sh_degree+1)**2 - 1, 3], shs,
                dtype=torch.float32, device="cuda").requires_grad_()

        # Setup geometry parameters
        if torch.is_tensor(density):
            if density.shape == (self.num_grid_pts, 1):
                self._geo_grid_pts = density.contiguous().cuda()
            elif density.shape == (self.num_voxels, 8):
                if reduce_density:
                    self._geo_grid_pts = torch.zeros(
                        [self.num_grid_pts, 1], dtype=torch.float32, device="cuda")
                    self._geo_grid_pts.index_reduce_(
                        dim=0,
                        index=self.vox_key.flatten(),
                        source=density.flatten(),
                        reduce="mean",
                        include_self=False)
                else:
                    self.frozen_vox_geo = density.contiguous().cuda()
            else:
                raise Exception(f"Unexpected density shape. "
                                f"It should be either {(self.num_grid_pts,1)} or {(self.num_voxels,8)}")
        else:
            self._geo_grid_pts = torch.full(
                [self.num_grid_pts, 1], density,
                dtype=torch.float32, device="cuda").requires_grad_()

    def ijkl_init(self,
                  scene_center,
                  scene_extent,
                  ijk,           # Nx3 integer coordinates of each voxel.
                  octlevel,      # Nx1 or scalar for the Octree level of each voxel.

                  # The following are model parameters.
                  # If the input are tensors, the gradient of rendering can be backprop to them.
                  # Otherwise, it creates new trainable tensors.
                  rgb=0.5,       # Nx3 or scalar for voxel color in range of 0~1.
                  shs=0.0,       # NxDx3 or scalar for voxel higher-deg sh coefficient.
                  density=-10.,  # Nx8 or Ngridx1 or scalar for voxel density field.
                                 # The order is [0,0,0] => [0,0,1] => [0,1,0] => [0,1,1] ...
                  reduce_density=False,  # Whether to merge grid points if density is Nx8.
                  ):

        scene_center, scene_extent, _ = get_scene_bound_tensor(
            center=scene_center, extent=scene_extent)

        # Convert to ijkl to octpath
        octlevel = get_octlevel_tensor(octlevel, num_voxels=len(ijk))

        assert torch.is_tensor(ijk)
        assert len(ijk.shape) == 2 and ijk.shape[1] == 3
        assert len(ijk) == len(octlevel)
        ijk = ijk.long()
        if (ijk < 0).any():
            raise Exception("xyz out of scene bound")
        if (ijk >= (1 << octlevel.long())).any():
            raise Exception("xyz out of scene bound")
        octpath = svraster_cuda.utils.ijk_2_octpath(ijk, octlevel)

        self.octpath_init(
            scene_center=scene_center,
            scene_extent=scene_extent,
            octpath=octpath,
            octlevel=octlevel,
            rgb=rgb,
            shs=shs,
            density=density,
            reduce_density=reduce_density)

    def points_init(self,
                         scene_center,
                         scene_extent,
                         xyz,           # Nx3 point coordinates in world space.
                         octlevel=None, # Nx1 or scalar for the Octree level of each voxel.
                         expected_vox_size=None,
                         level_round_mode='nearest',

                         # The following are model parameters.
                         # If the input are tensors, the gradient of rendering can be backprop to them.
                         # Otherwise, it creates new trainable tensors.
                         rgb=0.5,       # Nx3 or scalar for voxel color in range of 0~1.
                         shs=0.0,       # NxDx3 or scalar for voxel higher-deg sh coefficient.
                         density=-10.,  # Nx8 or scalar for voxel density field.
                                        # The order is [0,0,0] => [0,0,1] => [0,1,0] => [0,1,1] ...
                         reduce_density=False,  # Whether to merge grid points if density is Nx8.
                         ):

        scene_center, scene_extent, _ = get_scene_bound_tensor(center=scene_center, extent=scene_extent)

        # Compute voxel level
        if octlevel is not None:
            assert expected_vox_size is None
            octlevel = get_octlevel_tensor(octlevel, num_voxels=len(xyz))
        elif expected_vox_size is not None:
            octlevel_fp32 = octree_utils.vox_size_2_level(scene_extent, expected_vox_size)
            if level_round_mode == "nearest":
                octlevel_fp32 = octlevel_fp32.round()
            elif level_round_mode == "down":
                octlevel_fp32 = octlevel_fp32.floor()
            elif level_round_mode == "up":
                octlevel_fp32 = octlevel_fp32.ceil()
            else:
                raise Exception("Unknonw level_round_mode")
            octlevel_fp32 = octlevel_fp32.clamp(1, svraster_cuda.meta.MAX_NUM_LEVELS)
            octlevel = get_octlevel_tensor(octlevel_fp32.to(torch.int8), num_voxels=len(xyz))
        else:
            raise Exception("Either octlevel or expected_vox_size should be given.")

        # Transform point to ijk integer coordinate
        scene_min_xyz = scene_center - 0.5 * scene_extent
        vox_size = octree_utils.level_2_vox_size(scene_extent, octlevel)
        ijk = ((xyz - scene_min_xyz) / vox_size).long()

        # Reduce duplicated tensor
        ijkl = torch.cat([ijk, octlevel], dim=1)
        ijkl_unq, invmap = ijkl.unique(dim=0, return_inverse=True)
        ijk, octlevel = ijkl_unq.split([3, 1], dim=1)
        octlevel = octlevel.to(torch.int8)

        if torch.is_tensor(rgb):
            assert rgb.shape == (len(invmap), 3)
            new_shape = (len(ijk), 3)
            rgb = torch.zeros(new_shape, dtype=torch.float32, device="cuda").index_reduce_(
                dim=0,
                index=invmap,
                source=rgb,
                reduce="mean",
                include_self=False)

        if torch.is_tensor(shs):
            assert shs.shape == (len(invmap), (self.max_sh_degree+1)**2 - 1, 3)
            new_shape = (len(ijk), (self.max_sh_degree+1)**2 - 1, 3)
            shs = torch.zeros(new_shape, dtype=torch.float32, device="cuda").index_reduce_(
                dim=0,
                index=invmap,
                source=shs,
                reduce="mean",
                include_self=False)

        if torch.is_tensor(density):
            assert density.shape == (len(invmap), 8)
            new_shape = (len(ijk), 8)
            density = torch.zeros(new_shape, dtype=torch.float32, device="cuda").index_reduce_(
                dim=0,
                index=invmap,
                source=density,
                reduce="mean",
                include_self=False)

        # Allocate voxel using ijkl coordinate
        self.ijkl_init(
            scene_center=scene_center,
            scene_extent=scene_extent,
            ijk=ijk,
            octlevel=octlevel,
            rgb=rgb,
            shs=shs,
            density=density,
            reduce_density=reduce_density)


#################################################
# Helper function
#################################################
def get_scene_bound_tensor(center, extent, outside_level=0):
    if torch.is_tensor(center):
        scene_center = center.float().clone().cuda()
    else:
        scene_center = torch.tensor(center, dtype=torch.float32, device="cuda")

    if torch.is_tensor(extent):
        inside_extent = extent.float().clone().cuda()
    else:
        inside_extent = torch.tensor(extent, dtype=torch.float32, device="cuda")

    scene_extent = inside_extent * (2 ** outside_level)

    assert scene_center.shape == (3,)
    assert scene_extent.numel() == 1

    return scene_center, scene_extent, inside_extent

def get_octlevel_tensor(octlevel, num_voxels=None):
    if not torch.is_tensor(octlevel):
        assert np.all(octlevel > 0)
        assert np.all(octlevel <= svraster_cuda.meta.MAX_NUM_LEVELS)
        octlevel = torch.tensor(octlevel, dtype=torch.int8, device="cuda")
    if octlevel.numel() == 1:
        octlevel = octlevel.view(1, 1).repeat(num_voxels, 1).contiguous()
    octlevel = octlevel.reshape(-1, 1)
    assert octlevel.dtype == torch.int8
    assert num_voxels is None or octlevel.numel() == num_voxels

    return octlevel


#################################################
# Octree layout construction heuristic
#################################################
def octlayout_filtering(octpath, octlevel, scene_center, scene_extent, cameras=None, filter_zero_visiblity=True, filter_near=-1):

    vox_center, vox_size = octree_utils.octpath_decoding(
        octpath, octlevel,
        scene_center, scene_extent)

    # Filtering
    kept_mask = torch.ones([len(octpath)], dtype=torch.bool, device="cuda")
    if filter_zero_visiblity:
        assert cameras is not None, "Cameras should be given to filter invisible voxels"
        rate = svraster_cuda.renderer.mark_max_samp_rate(
            cameras, octpath, vox_center, vox_size)
        kept_mask &= (rate > 0)
    if filter_near > 0:
        is_near = svraster_cuda.renderer.mark_near(
            cameras, octpath, vox_center, vox_size, near=filter_near)
        kept_mask &= (~is_near)
    kept_idx = torch.where(kept_mask)[0]
    octpath = octpath[kept_idx]
    octlevel = octlevel[kept_idx]
    return octpath, octlevel


def octlayout_inside_uniform(scene_center, scene_extent, outside_level, n_level, cameras=None, filter_zero_visiblity=True, filter_near=-1):
    octpath, octlevel = octree_utils.gen_octpath_dense(
        outside_level=outside_level,
        n_level_inside=n_level)

    octpath, octlevel = octlayout_filtering(
        octpath=octpath,
        octlevel=octlevel,
        scene_center=scene_center,
        scene_extent=scene_extent,
        cameras=cameras,
        filter_zero_visiblity=filter_zero_visiblity,
        filter_near=filter_near)
    return octpath, octlevel


def octlayout_outside_heuristic(scene_center, scene_extent, outside_level, cameras, min_num, max_level, filter_near=-1):

    assert cameras is not None, "Cameras should provided in this mode."

    # Init by adding one sub-level in each shell level
    octpath = []
    octlevel = []
    for lv in range(1, 1+outside_level):
        path, lv = octree_utils.gen_octpath_shell(
            shell_level=lv,
            n_level_inside=1)
        octpath.append(path)
        octlevel.append(lv)
    octpath = torch.cat(octpath)
    octlevel = torch.cat(octlevel)

    # Iteratively subdivide voxels with maximum sampling rate
    while True:
        vox_center, vox_size = octree_utils.octpath_decoding(
            octpath, octlevel, scene_center, scene_extent)
        samp_rate = svraster_cuda.renderer.mark_max_samp_rate(
            cameras, octpath, vox_center, vox_size)

        kept_idx = torch.where((samp_rate > 0))[0]
        octpath = octpath[kept_idx]
        octlevel = octlevel[kept_idx]
        octlevel_mask = (octlevel.squeeze(1) < max_level)
        samp_rate = samp_rate[kept_idx] * octlevel_mask
        vox_size = vox_size[kept_idx]
        still_need_n = (min_num - len(octpath)) // 7
        still_need_n = min(len(octpath), round(still_need_n))
        if still_need_n <= 0:
            break
        rank = samp_rate * (octlevel.squeeze(1) < svraster_cuda.meta.MAX_NUM_LEVELS)
        subdiv_mask = (rank >= rank.sort().values[-still_need_n])
        subdiv_mask &= (octlevel.squeeze(1) < svraster_cuda.meta.MAX_NUM_LEVELS)
        subdiv_mask &= octlevel_mask
        samp_rate *= subdiv_mask
        subdiv_mask &= (samp_rate >= samp_rate.quantile(0.9))  # Subdivide only 10% each iteration
        if subdiv_mask.sum() == 0:
            break
        octpath_children, octlevel_children = octree_utils.gen_children(
            octpath[subdiv_mask], octlevel[subdiv_mask])
        octpath = torch.cat([octpath[~subdiv_mask], octpath_children])
        octlevel = torch.cat([octlevel[~subdiv_mask], octlevel_children])

    octpath, octlevel = octlayout_filtering(
        octpath=octpath,
        octlevel=octlevel,
        scene_center=scene_center,
        scene_extent=scene_extent,
        cameras=cameras,
        filter_zero_visiblity=True,
        filter_near=filter_near)
    return octpath, octlevel
