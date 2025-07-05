# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import json
import pycolmap
import numpy as np
from PIL import Image
from pathlib import Path
import concurrent.futures

from src.utils.colmap_utils import parse_colmap_pts
from src.utils.camera_utils import fov2focal, focal2fov


def read_nerf_dataset(source_path, test_every, use_test, camera_creator):

    source_path = Path(source_path)

    # Load training cameras
    if (source_path / "transforms_train.json").exists():
        train_cam_lst, point_cloud = read_cameras_from_json(
            source_path=source_path,
            meta_fname="transforms_train.json",
            camera_creator=camera_creator)
    else:
        train_cam_lst, point_cloud = read_cameras_from_json(
            source_path=source_path,
            meta_fname="transforms.json",
            camera_creator=camera_creator)

    # Load testing cameras
    if (source_path / "transforms_test.json").exists():
        test_cam_lst, _ = read_cameras_from_json(
            source_path=source_path,
            meta_fname="transforms_test.json",
            camera_creator=camera_creator)
    elif use_test:
        test_cam_lst = [
            cam for i, cam in enumerate(train_cam_lst)
            if i % test_every == 0]
        train_cam_lst = [
            cam for i, cam in enumerate(train_cam_lst)
            if i % test_every != 0]
    else:
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
        # Assume synthetic blender scene bound
        suggested_bounding = np.array([
            [-1.5, -1.5, -1.5],
            [1.5, 1.5, 1.5],
        ], dtype=np.float32)

    # Pack dataset
    dataset = {
        'train_cam_lst': train_cam_lst,
        'test_cam_lst': test_cam_lst,
        'suggested_bounding': suggested_bounding,
        'point_cloud': point_cloud,
    }
    return dataset


def read_cameras_from_json(source_path, meta_fname, camera_creator):

    with open(source_path / meta_fname) as f:
        meta = json.load(f)

    # Load COLMAP points if there is
    if "colmap" in meta:
        sfm = pycolmap.Reconstruction(source_path / meta["colmap"]["path"])
        if "transform" in meta["colmap"]:
            transform = np.array(meta["colmap"]["transform"])
        else:
            transform = None
        point_cloud = parse_colmap_pts(sfm, transform)
        correspondent = point_cloud.corr
    else:
        point_cloud = None
        correspondent = None

    # Load global setup
    global_fovx = meta.get("camera_angle_x", 0)
    global_fovy = meta.get("camera_angle_y", 0)
    global_cx_p = parse_principle_point(meta, is_cx=True)
    global_cy_p = parse_principle_point(meta, is_cx=False)

    # Load all images and cameras
    todo_lst = []
    for frame in meta["frames"]:

        # Guess the rgb image path and load image
        path_candidates = [
            source_path / frame["file_path"],
            source_path / (frame["file_path"] + '.png'),
            source_path / (frame["file_path"] + '.jpg'),
            source_path / (frame["file_path"] + '.JPG'),
        ]
        for image_path in path_candidates:
            if image_path.exists():
                break

        if frame.get('heldout', False):
            image = Image.new('RGB', (frame['w'], frame['h']))
        elif image_path.exists():
            image = Image.open(image_path)
        else:
            raise Exception(f"File not found: {str(image_path)}")

        # Load camera intrinsic
        fovx = frame.get('camera_angle_x', global_fovx)
        cx_p = frame.get('cx_p', global_cx_p)
        cy_p = frame.get('cy_p', global_cy_p)

        if 'camera_angle_y' in frame:
            fovy = frame['camera_angle_y']
        elif global_fovy > 0:
            fovy = global_fovy
        else:
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])

        # Load camera pose
        c2w = np.array(frame["transform_matrix"])
        c2w[:3, 1:3] *= -1  # from opengl y-up-z-back to colmap y-down-z-forward
        w2c = np.linalg.inv(c2w).astype(np.float32)

        # Load sparse point
        if point_cloud is not None:
            sparse_pt = point_cloud.points[correspondent[image_path.name]]
        else:
            sparse_pt = None

        todo_lst.append(dict(
            image=image,
            w2c=w2c,
            fovx=fovx,
            fovy=fovy,
            cx_p=cx_p,
            cy_p=cy_p,
            sparse_pt=sparse_pt,
            image_name=image_path.name,
        ))

    # Load all cameras concurrently
    import torch
    torch.inverse(torch.eye(3, device="cuda"))  # Fix module lazy loading bug:
                                                # https://github.com/pytorch/pytorch/issues/90613

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(camera_creator, **todo) for todo in todo_lst]
        cam_lst = [f.result() for f in futures]

    return cam_lst, point_cloud


def parse_principle_point(info, is_cx):
    key = "cx" if is_cx else "cy"
    key_res = "w" if is_cx else "h"
    if f"{key}_p" in info:
        return info[f"{key}_p"]
    if key in info and key_res in info:
        return info[key] / info[key_res]
    return None
