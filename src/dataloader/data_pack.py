# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import random
import numpy as np

import torch

from src.dataloader.reader_colmap_dataset import read_colmap_dataset
from src.dataloader.reader_nerf_dataset import read_nerf_dataset
from src.utils.camera_utils import interpolate_poses

from src.cameras import Camera, MiniCam


class DataPack:

    def __init__(self,
                 source_path,
                 image_dir_name="images",
                 mask_dir_name="masks",
                 res_downscale=0.,
                 res_width=0,
                 skip_blend_alpha=False,
                 alpha_is_white=False,
                 data_device="cpu",
                 use_test=False,
                 test_every=8,
                 camera_params_only=False):

        camera_creator = CameraCreator(
            res_downscale=res_downscale,
            res_width=res_width,
            skip_blend_alpha=skip_blend_alpha,
            alpha_is_white=alpha_is_white,
            data_device=data_device,
            camera_params_only=camera_params_only,
        )

        sparse_path = os.path.join(source_path, "sparse")
        colmap_path = os.path.join(source_path, "colmap", "sparse")
        meta_path1 = os.path.join(source_path, "transforms_train.json")
        meta_path2 = os.path.join(source_path, "transforms.json")

        # Read images concurrently
        s_time = time.perf_counter()

        if os.path.exists(sparse_path) or os.path.exists(colmap_path):
            print("Read dataset in COLMAP format.")
            dataset = read_colmap_dataset(
                source_path=source_path,
                image_dir_name=image_dir_name,
                mask_dir_name=mask_dir_name,
                use_test=use_test,
                test_every=test_every,
                camera_creator=camera_creator)
        elif os.path.exists(meta_path1) or os.path.exists(meta_path2):
            print("Read dataset in NeRF format.")
            dataset = read_nerf_dataset(
                source_path=source_path,
                use_test=use_test,
                test_every=test_every,
                camera_creator=camera_creator)
        else:
            raise Exception("Unknown scene type!")

        e_time = time.perf_counter()
        print(f"Read dataset in {e_time - s_time:.3f} seconds.")

        self._cameras = {
            'train': dataset['train_cam_lst'],
            'test': dataset['test_cam_lst'],
        }

        ##############################
        # Read additional dataset info
        ##############################
        # If the dataset suggested a scene bound
        self.suggested_bounding = dataset.get('suggested_bounding', None)

        # If the dataset provide a transformation to other coordinate
        self.to_world_matrix = None
        to_world_path = os.path.join(source_path, 'to_world_matrix.txt')
        if os.path.isfile(to_world_path):
            self.to_world_matrix = np.loadtxt(to_world_path)

        # If the dataset has a point cloud
        self.point_cloud = dataset.get('point_cloud', None)

    def get_train_cameras(self):
        return self._cameras['train']

    def get_test_cameras(self):
        return self._cameras['test']

    def interpolate_cameras(self, n_frames, starting_id=0, ids=[], step_forward=0):
        cams = self.get_train_cameras()
        if len(ids):
            key_poses = [cams[i].c2w.cpu().numpy() for i in ids]
        else:
            assert starting_id >= 0
            assert starting_id < len(cams)
            cam_pos = torch.stack([cam.position for cam in cams])
            ids = [starting_id]
            for _ in range(3):
                farthest_id = torch.cdist(cam_pos[ids], cam_pos).amin(0).argmax().item()
                ids.append(farthest_id)
            ids[1], ids[2] = ids[2], ids[1]
            key_poses = [cams[i].c2w.cpu().numpy() for i in ids]

        if step_forward != 0:
            for i in range(len(key_poses)):
                lookat = key_poses[i][:3, 2]
                key_poses[i][:3, 3] += step_forward * lookat

        interp_poses = interpolate_poses(key_poses, n_frame=n_frames, periodic=True)

        base_cam = cams[ids[0]]
        interp_cams = [
            MiniCam(
                c2w=pose,
                fovx=base_cam.fovx, fovy=base_cam.fovy,
                width=base_cam.image_width, height=base_cam.image_height)
            for pose in interp_poses]
        return interp_cams


# Create a random sequence of image indices
def compute_iter_idx(num_data, num_iter):
    tr_iter_idx = []
    while len(tr_iter_idx) < num_iter:
        lst = list(range(num_data))
        random.shuffle(lst)
        tr_iter_idx.extend(lst)
    return tr_iter_idx[:num_iter]


# Function that create Camera instances while parsing dataset
class CameraCreator:

    warned = False

    def __init__(self,
                 res_downscale=0.,
                 res_width=0,
                 skip_blend_alpha=False,
                 alpha_is_white=False,
                 data_device="cpu",
                 camera_params_only=False):

        self.res_downscale = res_downscale
        self.res_width = res_width
        self.skip_blend_alpha = skip_blend_alpha
        self.alpha_is_white = alpha_is_white
        self.data_device = data_device
        self.camera_params_only = camera_params_only

    def __call__(self,
                 image,
                 w2c,
                 fovx,
                 fovy,
                 cx_p=0.5,
                 cy_p=0.5,
                 sparse_pt=None,
                 image_name="",
                 mask=None):

        # Determine target resolution
        if self.res_downscale > 0:
            downscale = self.res_downscale
        elif self.res_width > 0:
            downscale = image.size[0] / self.res_width
        else:
            downscale = 1

            total_pix = image.size[0] * image.size[1]
            if total_pix > 1200 ** 2 and not self.warned:
                self.warned = True
                suggest_ds = (total_pix ** 0.5) / 1200
                print(f"###################################################################")
                print(f"Image too large. Suggest to use `--res_downscale {suggest_ds:.1f}`.")
                print(f"###################################################################")

        # Load camera parameters only
        if self.camera_params_only:
            return MiniCam(
                c2w=np.linalg.inv(w2c),
                fovx=fovx, fovy=fovy,
                cx_p=cx_p, cy_p=cy_p,
                width=round(image.size[0] / downscale),
                height=round(image.size[1] / downscale),
                image_name=image_name)

        # Resize image if needed
        if downscale != 1:
            size = (round(image.size[0] / downscale), round(image.size[1] / downscale))
            image = image.resize(size)

        # Convert image to tensor
        tensor = torch.tensor(np.array(image), dtype=torch.float32).moveaxis(-1, 0) / 255.0
        if tensor.shape[0] == 4:
            # Blend alpha channel
            tensor, mask = tensor.split([3, 1], dim=0)
            if not self.skip_blend_alpha:
                tensor = tensor * mask + int(self.alpha_is_white) * (1 - mask)

        # Conver mask to tensor if there is
        if mask is not None:
            size = tensor.shape[-2:][::-1]
            if mask.size != size:
                mask = mask.resize(size)
            mask = torch.tensor(np.array(mask), dtype=torch.float32) / 255.0
            if len(mask.shape) == 3:
                mask = mask.mean(-1)
            mask = mask[None]

        return Camera(
            w2c=w2c,
            fovx=fovx, fovy=fovy,
            cx_p=cx_p, cy_p=cy_p,
            image=tensor,
            mask=mask,
            sparse_pt=sparse_pt,
            image_name=image_name)
