# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import json
import natsort
import pycolmap
import numpy as np
from PIL import Image
from pathlib import Path
import concurrent.futures

from src.utils.colmap_utils import parse_colmap_pts
from src.utils.camera_utils import focal2fov


def read_colmap_dataset(source_path, image_dir_name, mask_dir_name, use_test, test_every, camera_creator):

    source_path = Path(source_path)

    # Parse colmap meta data
    sparse_path = source_path / "sparse" / "0"
    if not sparse_path.exists():
        sparse_path = source_path / "colmap" / "sparse" / "0"
    if not sparse_path.exists():
        raise Exception("Can not find COLMAP reconstruction.")

    sfm = pycolmap.Reconstruction(sparse_path)
    point_cloud = parse_colmap_pts(sfm)
    correspondent = point_cloud.corr

    # Sort key by filename
    keys = natsort.natsorted(
        sfm.images.keys(),
        key = lambda k : sfm.images[k].name)

    # Load all images and cameras
    todo_lst = []
    for key in keys:

        frame = sfm.images[key]

        # Load image
        image_path = source_path / image_dir_name / frame.name
        if not image_path.exists():
            image_path = image_path.with_suffix('.png')
        if not image_path.exists():
            image_path = image_path.with_suffix('.jpg')
        if not image_path.exists():
            image_path = image_path.with_suffix('.JPG')
        if not image_path.exists():
            raise Exception(f"File not found: {str(image_path)}")
        image = Image.open(image_path)

        # Load camera intrinsic
        if frame.camera.model.name == "SIMPLE_PINHOLE":
            focal_x, cx, cy = frame.camera.params
            fovx = focal2fov(focal_x, frame.camera.width)
            fovy = focal2fov(focal_x, frame.camera.height)
            cx_p = cx / frame.camera.width
            cy_p = cy / frame.camera.height
        elif frame.camera.model.name == "PINHOLE":
            focal_x, focal_y, cx, cy = frame.camera.params
            fovx = focal2fov(focal_x, frame.camera.width)
            fovy = focal2fov(focal_y, frame.camera.height)
            cx_p = cx / frame.camera.width
            cy_p = cy / frame.camera.height
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # Load camera extrinsic
        w2c = np.eye(4, dtype=np.float32)
        try:
            w2c[:3] = frame.cam_from_world().matrix()
        except:
            # Older version of pycolmap
            w2c[:3] = frame.cam_from_world.matrix()

        # Load sparse point
        sparse_pt = point_cloud.points[correspondent[frame.name]]

        # Load mask if there is
        mask_path = (source_path / mask_dir_name / frame.name).with_suffix('.png')
        if mask_path.exists():
            mask = Image.open(mask_path)
        else:
            mask = None

        todo_lst.append(dict(
            image=image,
            w2c=w2c,
            fovx=fovx,
            fovy=fovy,
            cx_p=cx_p,
            cy_p=cy_p,
            sparse_pt=sparse_pt,
            image_name=image_path.name,
            mask=mask,
        ))

    # Load all cameras concurrently
    import torch
    torch.inverse(torch.eye(3, device="cuda"))  # Fix module lazy loading bug:
                                                # https://github.com/pytorch/pytorch/issues/90613

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(camera_creator, **todo) for todo in todo_lst]
        cam_lst = [f.result() for f in futures]

    # Split train/test
    if use_test:
        train_cam_lst = [
            cam for i, cam in enumerate(cam_lst)
            if i % test_every != 0]
        test_cam_lst = [
            cam for i, cam in enumerate(cam_lst)
            if i % test_every == 0]
    else:
        train_cam_lst = cam_lst
        test_cam_lst = []

    # Parse main scene bound if there is
    nerf_normalization_path = os.path.join(source_path, "nerf_normalization.json")
    if os.path.isfile(nerf_normalization_path):
        with open(nerf_normalization_path) as f:
            nerf_normalization = json.load(f)
        suggested_center = np.array(nerf_normalization["center"], dtype=np.float32)
        suggested_radius = np.array(nerf_normalization["radius"], dtype=np.float32)
        suggested_bounding = np.stack([
            suggested_center - suggested_radius,
            suggested_center + suggested_radius,
        ])
    else:
        suggested_bounding = None

    # Pack dataset
    dataset = {
        'train_cam_lst': train_cam_lst,
        'test_cam_lst': test_cam_lst,
        'suggested_bounding': suggested_bounding,
        'point_cloud': point_cloud,
    }
    return dataset
