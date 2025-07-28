# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#TEST
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#TENSORBOARD
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # the number is the gpu number as shown in nvidia-smi.
import json
import time
import contextlib
import uuid
import imageio
import datetime
import numpy as np
from tqdm import tqdm

import torch

from src.config import cfg, update_argparser, update_config

from src.utils.system_utils import seed_everything
from src.utils.image_utils import im_tensor2np, viz_tensordepth
from src.utils.bounding_utils import decide_main_bounding
from src.utils import mono_utils
from src.utils import loss_utils

from src.dataloader.data_pack import DataPack, compute_iter_idx
from src.sparse_voxel_model import SparseVoxelModel

import svraster_cuda
import yaml
import sys
sys.path.append("/home/UFAD/darrenlau/nerfbusters/")

from nerfbusters.nerf.nerfbusters_pipeline import NerfbustersPipelineConfig, NerfbustersPipeline
from dotmap import DotMap

# Alpha thresholding parameters (tune these for your scene)
ALPHA_THRESHOLD_CONFIG = {
    'start_iteration': 2000,        # When to start applying
    'frequency': 100,               # Apply every N iterations  
    'base_threshold': 0.1,         # Base difference threshold
    'cube_size': 16,                # Cube resolution (16³ = 4096 voxels)
    'num_cubes': 10,                 # Number of cubes per iteration
    'lambda_min': 0.1,              # Minimum diffusion weight
    'lambda_max': 0.3,              # Maximum diffusion weight
    'target_psnr': 26.0             # Target PSNR for adaptive weighting
}

# Add timing utility class
class Timer:
    def __init__(self):
        self.times = {}
    
    @contextlib.contextmanager
    def time(self, name):
        start = time.time()
        yield
        elapsed = time.time() - start
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed)
    
    def report(self, iteration):
        if iteration % 100 == 0:  # Report every 100 iterations
            print(f"\n[TIMING REPORT] Iteration {iteration}")
            print("=" * 60)
            for name, times_list in self.times.items():
                if times_list:  # Only show if we have data
                    avg_time = sum(times_list) / len(times_list)
                    total_time = sum(times_list)
                    print(f"{name:30s}: {avg_time*1000:6.1f}ms avg, {total_time:6.1f}s total ({len(times_list)} calls)")
            print("=" * 60)
            # Clear times after reporting
            self.times.clear()

timer = Timer()


def training(args):

    # Init and load data pack
    data_pack = DataPack(
        source_path=cfg.data.source_path,
        image_dir_name=cfg.data.image_dir_name,
        res_downscale=cfg.data.res_downscale,
        res_width=cfg.data.res_width,
        skip_blend_alpha=cfg.data.skip_blend_alpha,
        alpha_is_white=cfg.model.white_background,
        data_device=cfg.data.data_device,
        use_test=cfg.data.eval,
        test_every=cfg.data.test_every,
    )

    diff_config_path = "/home/UFAD/darrenlau/nerfbusters/config/shapenet.yaml"
    diff_ckpt_path = "/home/UFAD/darrenlau/nerfbusters/data/nerfbusters-diffusion-cube-weights.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion_model = load_diffusion_model(diff_config_path, diff_ckpt_path, device)
    print("✅ Diffusion model loaded:", type(diffusion_model))

    weight_grid = SVRWeightGrid(resolution = 64, device = device)
    print("✅ Weight grid initialized")

    # Instantiate data loader
    tr_cams = data_pack.get_train_cameras()
    tr_cam_indices = compute_iter_idx(len(tr_cams), cfg.procedure.n_iter)

    if cfg.auto_exposure.enable:
        for cam in tr_cams:
            cam.auto_exposure_init()

    # Prepare monocular depth priors if instructed
    if cfg.regularizer.lambda_depthanythingv2:
        mono_utils.prepare_depthanythingv2(
            cameras=tr_cams,
            source_path=cfg.data.source_path,
            force_rerun=False)

    if cfg.regularizer.lambda_mast3r_metric_depth:
        mono_utils.prepare_mast3r_metric_depth(
            cameras=tr_cams,
            source_path=cfg.data.source_path,
            mast3r_repo_path=cfg.regularizer.mast3r_repo_path)

    # Decide main (inside) region bounding box
    bounding = decide_main_bounding(
        bound_mode=cfg.bounding.bound_mode,
        forward_dist_scale=cfg.bounding.forward_dist_scale,
        pcd_density_rate=cfg.bounding.pcd_density_rate,
        bound_scale=cfg.bounding.bound_scale,
        tr_cams=tr_cams,
        pcd=data_pack.point_cloud,
        suggested_bounding=data_pack.suggested_bounding)

    # Init voxel model
    voxel_model = SparseVoxelModel(
        n_samp_per_vox=cfg.model.n_samp_per_vox,
        sh_degree=cfg.model.sh_degree,
        ss=cfg.model.ss,
        white_background=cfg.model.white_background,
        black_background=cfg.model.black_background,
    )

    if args.load_iteration:
        loaded_iter = voxel_model.load_iteration(
            args.model_path, args.load_iteration)
    else:
        loaded_iter = None
        voxel_model.model_init(
            bounding=bounding,
            outside_level=cfg.bounding.outside_level,
            init_n_level=cfg.init.init_n_level,
            init_out_ratio=cfg.init.init_out_ratio,
            sh_degree_init=cfg.init.sh_degree_init,
            geo_init=cfg.init.geo_init,
            sh0_init=cfg.init.sh0_init,
            shs_init=cfg.init.shs_init,
            cameras=tr_cams,
        )

        if not test_nerfbusters_integration(voxel_model, diffusion_model, weight_grid):
            print("⚠️ Integration test failed - check your setup!")
        else:
            print("🚀 Nerfbusters integration ready!")


    first_iter = loaded_iter if loaded_iter else 1
    print(f"Start optmization from iters={first_iter}.")

    # Init optimizer
    def create_trainer():
        # The pytorch built-in `torch.optim.Adam` also works
        optimizer = svraster_cuda.sparse_adam.SparseAdam(
            [
                #Voxel_model._geo_grid_pts is the density field
                {'params': [voxel_model._geo_grid_pts], 'lr': cfg.optimizer.geo_lr},
                {'params': [voxel_model._sh0], 'lr': cfg.optimizer.sh0_lr},
                {'params': [voxel_model._shs], 'lr': cfg.optimizer.shs_lr},
            ],
            betas=(cfg.optimizer.optim_beta1, cfg.optimizer.optim_beta2),
            eps=cfg.optimizer.optim_eps)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.optimizer.lr_decay_ckpt,
            gamma=cfg.optimizer.lr_decay_mult)
        return optimizer, scheduler

    optimizer, scheduler = create_trainer()
    if not loaded_iter:
        checkpoint_data = load_checkpoint(voxel_model, optimizer, scheduler, args.model_path)
        if checkpoint_data:
            first_iter = checkpoint_data['iteration'] + 1
            ema_loss_for_log = checkpoint_data['ema_loss']
            ema_psnr_for_log = checkpoint_data['ema_psnr']
            print(f"Resuming from checkpoint at iteration {first_iter-1}")
    elif args.load_optimizer:
        # Your existing optimizer loading code
        optim_ckpt = torch.load(os.path.join(args.model_path, "optim.pt"))
        optimizer.load_state_dict(optim_ckpt['optim'])
        scheduler.load_state_dict(optim_ckpt['sched'])
        del optim_ckpt

    # Some other initialization
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    elapsed = 0

    tr_render_opt = {
        'track_max_w': False,
        'lambda_R_concen': cfg.regularizer.lambda_R_concen,
        'output_T': False,
        'output_depth': False,
        'ss': 1.0,  # disable supersampling at first
        'rand_bg': cfg.regularizer.rand_bg,
        'use_auto_exposure': cfg.auto_exposure.enable,
    }

    sparse_depth_loss = loss_utils.SparseDepthLoss(
        iter_end=cfg.regularizer.sparse_depth_until)
    depthanythingv2_loss = loss_utils.DepthAnythingv2Loss(
        iter_from=cfg.regularizer.depthanythingv2_from,
        iter_end=cfg.regularizer.depthanythingv2_end,
        end_mult=cfg.regularizer.depthanythingv2_end_mult)
    mast3r_metric_depth_loss = loss_utils.Mast3rMetricDepthLoss(
        iter_from=cfg.regularizer.mast3r_metric_depth_from,
        iter_end=cfg.regularizer.mast3r_metric_depth_end,
        end_mult=cfg.regularizer.mast3r_metric_depth_end_mult)
    nd_loss = loss_utils.NormalDepthConsistencyLoss(
        iter_from=cfg.regularizer.n_dmean_from,
        iter_end=cfg.regularizer.n_dmean_end,
        ks=cfg.regularizer.n_dmean_ks,
        tol_deg=cfg.regularizer.n_dmean_tol_deg)
    nmed_loss = loss_utils.NormalMedianConsistencyLoss(
        iter_from=cfg.regularizer.n_dmed_from,
        iter_end=cfg.regularizer.n_dmed_end)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    iter_rng = range(first_iter, cfg.procedure.n_iter+1)
    progress_bar = tqdm(iter_rng, desc="Training")

    for iteration in iter_rng:

        # Start processing time tracking of this iteration
        iter_start.record()

        # Increase the degree of SH by one up to a maximum degree
        if iteration % 1000 == 0:
            voxel_model.sh_degree_add1()

        # Recompute sh from cameras
        if iteration in cfg.procedure.reset_sh_ckpt:
            print("Reset sh0 from cameras.")
            print("Reset shs to zero.")
            voxel_model.reset_sh_from_cameras(tr_cams)
            torch.cuda.empty_cache()

        # Use default super-sampling option
        if iteration > 1000:
            if cfg.regularizer.ss_aug_max > 1:
                tr_render_opt['ss'] = np.random.uniform(1, cfg.regularizer.ss_aug_max)
            elif 'ss' in tr_render_opt:
                tr_render_opt.pop('ss')  # Use default ss

        need_sparse_depth = cfg.regularizer.lambda_sparse_depth > 0 and sparse_depth_loss.is_active(iteration)
        need_depthanythingv2 = cfg.regularizer.lambda_depthanythingv2 > 0 and depthanythingv2_loss.is_active(iteration)
        need_mast3r_metric_depth = cfg.regularizer.lambda_mast3r_metric_depth > 0 and mast3r_metric_depth_loss.is_active(iteration)
        need_nd_loss = cfg.regularizer.lambda_normal_dmean > 0 and nd_loss.is_active(iteration)
        need_nmed_loss = cfg.regularizer.lambda_normal_dmed > 0 and nmed_loss.is_active(iteration)
        tr_render_opt['output_T'] = cfg.regularizer.lambda_T_concen > 0 or cfg.regularizer.lambda_T_inside > 0 or cfg.regularizer.lambda_mask > 0 or need_sparse_depth or need_nd_loss or need_depthanythingv2 or need_mast3r_metric_depth
        tr_render_opt['output_normal'] = need_nd_loss or need_nmed_loss
        tr_render_opt['output_depth'] = need_sparse_depth or need_nd_loss or need_nmed_loss or need_depthanythingv2 or need_mast3r_metric_depth

        if iteration >= cfg.regularizer.dist_from and cfg.regularizer.lambda_dist:
            tr_render_opt['lambda_dist'] = cfg.regularizer.lambda_dist

        if iteration >= cfg.regularizer.ascending_from and cfg.regularizer.lambda_ascending:
            tr_render_opt['lambda_ascending'] = cfg.regularizer.lambda_ascending

        # Update auto exposure
        if cfg.auto_exposure.enable and iteration in cfg.procedure.auto_exposure_upd_ckpt:
            for cam in tr_cams:
                with torch.no_grad():
                    ref = voxel_model.render(cam, ss=1.0)['color']
                cam.auto_exposure_update(ref, cam.image.cuda())

        # Pick a Camera
        cam = tr_cams[tr_cam_indices[iteration-1]]

        #Get gt image
        #IN PROGRESS - get GT density
        gt_image = cam.image.cuda()
        if cfg.regularizer.lambda_R_concen > 0:
            tr_render_opt['gt_color'] = gt_image

        # Render
        #this is where the pixels are actually rendered with color and transparency
        #IN PROGRESS - Does it render the density values?
        with timer.time("TIME - NeRF Rendering"):
            render_pkg = voxel_model.render(cam, **tr_render_opt)
            render_image = render_pkg['color']

        mse = loss_utils.l2_loss(render_image, gt_image)

        if cfg.regularizer.use_l1:
            photo_loss = loss_utils.l1_loss(render_image, gt_image)
        elif cfg.regularizer.use_huber:
            photo_loss = loss_utils.huber_loss(render_image, gt_image, cfg.regularizer.huber_thres)
        else:
            photo_loss = mse
        loss = cfg.regularizer.lambda_photo * photo_loss

        with timer.time("TIME - Grabbing Density Field"):
            density_field = voxel_model._geo_grid_pts # [N_voxels, 1]

        #IN PROGRESS - Implementing New Loss Function
        #DIFFUSION LOSS COMPUTATION
        '''OLD LOSS LOOP - GRADIENTS DO NOT FLOW!
        L_diff = 0.0
        if iteration % 10 == 0 and iteration > 1000:  # Start after some warmup
            try:
                original_density_cubes = octree_to_regular_cubes(voxel_model, cube_size=32, num_cubes=10)
                
                with torch.no_grad():  # Remove if want gradients through diffusion
                    unet_model = diffusion_model.model
                    timestep = torch.tensor([10], device=original_density_cubes.device)
                    diffusion_output = unet_model(original_density_cubes, timestep)
                    enhanced_density_cubes = diffusion_output.sample
                
                L_diff = torch.nn.functional.mse_loss(enhanced_density_cubes, original_density_cubes)

                if iteration % 100 == 0:
                    print(f"[DIFFUSION] L_diff: {L_diff.item():.6f}")
                    
            except Exception as e:
                print(f"Diffusion loss computation failed: {e}")
                L_diff = 0.0

        mse = loss_utils.l2_loss(render_image, gt_image)
        if cfg.regularizer.use_l1:
            photo_loss = loss_utils.l1_loss(render_image, gt_image)
        elif cfg.regularizer.use_huber:
            photo_loss = loss_utils.huber_loss(render_image, gt_image, cfg.regularizer.huber_thres)
        else:
            photo_loss = mse

        # COMBINED LOSS: L = L_RGB + λ * L_diff
        L_RGB = cfg.regularizer.lambda_photo * photo_loss
        lambda_diff = 0.5  
        L_total = L_RGB + lambda_diff * L_diff

        loss = L_total

        if iteration % 100 == 0:
            print(f"[LOSS] RGB: {L_RGB.item():.6f}, Diff: {(lambda_diff * L_diff):.6f}, Total: {L_total.item():.6f}")
        '''

        #NERFBUSTERS DIFFUSION LOSS WITH IMPORTANCE SAMPLING------------------------------------------------
        '''
        L_diff = 0.0
        if iteration > 500 and iteration % 50 == 0:  # Start after warmup, apply every 5 iterations
            try:
                with timer.time("TIME - NB Importance Sampling"):
                    nerfbusters_loss, xyz, scales = nerfbusters_importance_sampling(
                        voxel_model, diffusion_model, weight_grid, iteration, num_cubes=10
                    )
                L_diff = nerfbusters_loss
                
                if iteration % 100 == 0:
                    print(f"[NERFBUSTERS] Loss: {L_diff.item():.6f}")
                    print(f"[NERFBUSTERS] Sampled {xyz.shape[0]} cubes of size {xyz.shape[1:]}") 
                    
            except Exception as e:
                print(f"Nerfbusters integration failed: {e}")
                L_diff = torch.tensor(0.0, device=voxel_model._geo_grid_pts.device)

        # COMBINED LOSS: L = L_RGB + λ * L_diff  
        L_RGB = cfg.regularizer.lambda_photo * photo_loss
        lambda_diff = 0.2
        L_total = L_RGB + lambda_diff * L_diff
        loss = L_total

        if iteration % 100 == 0 and L_diff > 0:
            print(f"[LOSS] RGB: {L_RGB.item():.6f}, Nerfbusters: {(lambda_diff * L_diff).item():.6f}, Total: {L_total.item():.6f}")

        if iteration % 500 == 0 and iteration > 500:
            try:
                # Check weight grid statistics
                with timer.time("TIME - Reporting Weight Grid"):
                    total_weights = weight_grid._weights.sum().item()
                    max_weight = weight_grid._weights.max().item()
                    print(f"[WEIGHT_GRID] Total: {total_weights:.2f}, Max: {max_weight:.2f}")
                
                # Sample and show importance regions
                with timer.time("TIME - Reporting Sample Importance Centers"):
                    centers = weight_grid.sample_importance_centers(10)
                    if weight_grid.scene_bounds:
                        min_b, max_b = weight_grid.scene_bounds
                        world_centers = centers * (max_b - min_b) + min_b
                        print(f"[SAMPLING] Centers range: [{world_centers.min():.2f}, {world_centers.max():.2f}]")
                    
            except Exception as e:
                print(f"Monitoring failed: {e}")'''
        #----------------------------------------------------------------------------------------------
        '''
        #THRESHOLDING LOSS-----------------------------------------------------------------------------
        L_diff = 0.0
        avg_diff_score = 0.0
        if iteration > 100 and iteration % 100 == 0:  # Start later, less frequent
            try:
                # Adaptive threshold based on training progress
                current_threshold = adaptive_threshold_schedule(iteration, ema_psnr_for_log)
                
                # Apply alpha thresholding diffusion WITH gradient flow
                alpha_loss, diff_score = alpha_thresholding_diffusion_with_gradients(
                    voxel_model, diffusion_model, cam, iteration,
                    threshold=current_threshold,
                    cube_size=16,  # Smaller for speed
                    num_cubes=6    # Fewer cubes for speed
                )
                
                L_diff = alpha_loss
                avg_diff_score = diff_score
                
                if iteration % 200 == 0:
                    print(f"[ALPHA-DIFFUSION] Threshold: {current_threshold:.4f}")
                    print(f"[ALPHA-DIFFUSION] Avg diff score: {avg_diff_score:.4f}")
                    print(f"[ALPHA-DIFFUSION] Loss: {L_diff.item():.6f}")
                    print(f"[ALPHA-DIFFUSION] Loss requires_grad: {L_diff.requires_grad}")  # Should be True
                            
            except Exception as e:
                print(f"Alpha thresholding diffusion failed: {e}")
                L_diff = torch.tensor(0.0, device=voxel_model._geo_grid_pts.device)
        #-----------------------------------------------------------------------------------------------------

        # COMBINED LOSS with adaptive weighting
        L_RGB = cfg.regularizer.lambda_photo * photo_loss

        # Adaptive lambda based on PSNR progress
        if ema_psnr_for_log < 24.0:
            lambda_diff = 0.1  # Focus on reconstruction first
        elif ema_psnr_for_log < 26.0:
            lambda_diff = 0.2  # Balanced approach
        else:
            lambda_diff = 0.3  # Strong regularization for fine details

        L_total = L_RGB + lambda_diff * L_diff
        loss = L_total

        # Enhanced logging
        if iteration % 100 == 0 and L_diff > 0:
            print(f"[LOSS] RGB: {L_RGB.item():.6f}, Alpha-Diff: {(lambda_diff * L_diff).item():.6f}, Total: {L_total.item():.6f}")
            print(f"[METRICS] PSNR: {ema_psnr_for_log:.2f}, Lambda: {lambda_diff:.2f}")

    
                # Enhanced monitoring for alpha thresholding
        if iteration % 500 == 0 and iteration > 2000:
            try:
                # Test alpha thresholding effectiveness
                with torch.no_grad():
                    test_cubes, _, _ = extract_view_relevant_cubes(voxel_model, cam, cube_size=16, num_cubes=3)
                    enhanced_test = diffusion_model.model(test_cubes, torch.tensor([10], device=test_cubes.device)).sample
                    
                    test_diff_scores, test_confidence, _ = compute_cube_differences(test_cubes, enhanced_test)
                    
                    print(f"[MONITORING] Test confidence range: [{test_confidence.min():.3f}, {test_confidence.max():.3f}]")
                    print(f"[MONITORING] Test diff range: [{test_diff_scores.min():.3f}, {test_diff_scores.max():.3f}]")
                    
                    # Check if we're making progress toward PSNR target
                    if ema_psnr_for_log > 25.0:
                        print(f"🎯 APPROACHING TARGET! Current PSNR: {ema_psnr_for_log:.2f}")
                    elif iteration > 10000:
                        print(f"⚠️ Progress check: PSNR {ema_psnr_for_log:.2f} at iter {iteration}")
                        
            except Exception as e:
                print(f"Monitoring failed: {e}")

        #-----------------------------------------------------------------------------------------------------
        '''

        #---------------------------------RANDOMS SAMPLING---------------------------------------
        L_diff = 0.0
        if iteration > 500 and iteration % 100 == 0:
            try:
                with timer.time("TIME - NB Importance Sampling"):
                    nerfbusters_loss, xyz, scales = nerfbusters_random_sampling(
                        voxel_model, diffusion_model, iteration, num_cubes=10
                    )
                L_diff = nerfbusters_loss
                
                if iteration % 100 == 0:
                    print(f"[NERFBUSTERS] Loss: {L_diff.item():.6f}")
                    print(f"[NERFBUSTERS] Sampled {xyz.shape[0]} cubes of size {xyz.shape[1:]}")
            except Exception as e:
                print(f"Nerfbusters integration failed: {e}")
                L_diff = torch.tensor(0.0, device=voxel_model._geo_grid_pts.device)

        L_RGB = cfg.regularizer.lambda_photo * photo_loss
        lambda_diff = 25
        L_total = L_RGB + lambda_diff * L_diff

        loss = L_total

        if iteration % 100 == 0 and L_diff > 0:
            print(f"[LOSS] RGB: {L_RGB.item():.6f}, Nerfbusters: {(lambda_diff * L_diff).item():.6f}, Total: {L_total.item():.6f}")

        #----------------------------------------------------------------------------------------------

        # NEW THRESHOLDING APPßROACH'
        '''
        if iteration > 100 and iteration % 50 == 0:
            try:
                # Use adaptive threshold
                current_threshold = adaptive_threshold_schedule(iteration, ema_psnr_for_log)
                
                with timer.time("TIME - Density Thresholding"):
                    apply_thresholding_to_density_field(
                        voxel_model, diffusion_model, iteration, 
                        threshold=current_threshold, 
                        num_cubes=10, 
                        attenuation_factor=0.05  # Reduce density by 30% in noisy regions
                    )
            except Exception as e:
                print(f"Density thresholding failed: {e}")

        # Remove L_diff from total loss - just use RGB loss
        loss = cfg.regularizer.lambda_photo * photo_loss
        '''
        if need_sparse_depth:
            loss += cfg.regularizer.lambda_sparse_depth * sparse_depth_loss(cam, render_pkg)

        if cfg.regularizer.lambda_mask:
            gt_T = 1 - cam.mask.cuda()
            loss += cfg.regularizer.lambda_mask * loss_utils.l2_loss(render_pkg['T'], gt_T)

        if need_depthanythingv2:
            loss += cfg.regularizer.lambda_depthanythingv2 * depthanythingv2_loss(cam, render_pkg, iteration)

        if need_mast3r_metric_depth:
            loss += cfg.regularizer.lambda_mast3r_metric_depth * mast3r_metric_depth_loss(cam, render_pkg, iteration)

        if cfg.regularizer.lambda_ssim:
            loss += cfg.regularizer.lambda_ssim * loss_utils.fast_ssim_loss(render_image, gt_image)
        if cfg.regularizer.lambda_T_concen:
            loss += cfg.regularizer.lambda_T_concen * loss_utils.prob_concen_loss(render_pkg[f'raw_T'])
        if cfg.regularizer.lambda_T_inside:
            loss += cfg.regularizer.lambda_T_inside * render_pkg[f'raw_T'].square().mean()
        if need_nd_loss:
            loss += cfg.regularizer.lambda_normal_dmean * nd_loss(cam, render_pkg, iteration)
        if need_nmed_loss:
            loss += cfg.regularizer.lambda_normal_dmed * nmed_loss(cam, render_pkg, iteration)

        # Backward to get gradient of current iteration
        optimizer.zero_grad(set_to_none=True)
        with timer.time("TIMER - Backwards Prop"):
            loss.backward()

        # Total variation regularization
        if cfg.regularizer.lambda_tv_density and \
                iteration >= cfg.regularizer.tv_from and \
                iteration <= cfg.regularizer.tv_until:
            voxel_model.apply_tv_on_density_field(cfg.regularizer.lambda_tv_density)

        # Optimizer step
        optimizer.step()
        if iteration % 1000 == 0 or iteration == cfg.procedure.n_iter:
            save_checkpoint(voxel_model, optimizer, scheduler, iteration, args.model_path, ema_loss_for_log, ema_psnr_for_log)

        ######################################################
        # Start adaptive voxels pruning and subdividing
        ######################################################

        meet_adapt_period = (
            iteration % cfg.procedure.adapt_every == 0 and \
            iteration >= cfg.procedure.adapt_from and \
            iteration <= cfg.procedure.n_iter-500)
        need_pruning = (
            meet_adapt_period and \
            iteration <= cfg.procedure.prune_until)
        need_subdividing = (
            meet_adapt_period and \
            iteration <= cfg.procedure.subdivide_until and \
            voxel_model.num_voxels < cfg.procedure.subdivide_max_num)

        if need_pruning or need_subdividing:
            # Track voxel statistic
            stat_pkg = voxel_model.compute_training_stat(camera_lst=tr_cams)
            # Cache scheduler state
            scheduler_state = scheduler.state_dict()

        if need_pruning:
            ori_n = voxel_model.num_voxels

            # Compute pruning threshold
            prune_thres = np.interp(
                iteration,
                xp=[cfg.procedure.adapt_from, cfg.procedure.prune_until],
                fp=[cfg.procedure.prune_thres_init, cfg.procedure.prune_thres_final])

            # Prune voxels
            prune_mask = (stat_pkg['max_w'] < prune_thres).squeeze(1)

            # Pruning
            voxel_model.pruning(prune_mask)

            # Show statistic
            new_n = voxel_model.num_voxels
            print(f'[PRUNING]     {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f};  thres={prune_thres:.4f})')

        if need_subdividing:
            ori_n = voxel_model.num_voxels

            # Exclude some voxels
            min_samp_interval = stat_pkg['min_samp_interval']
            if need_pruning:
                min_samp_interval = min_samp_interval[~prune_mask]
            size_thres = min_samp_interval * cfg.procedure.subdivide_samp_thres
            large_enough = (voxel_model.vox_size * 0.5 > size_thres).squeeze(1)
            non_finest = voxel_model.octlevel.squeeze(1) < svraster_cuda.meta.MAX_NUM_LEVELS
            valid_mask = large_enough & non_finest

            # Compute subdivision threshold
            priority = voxel_model.subdivision_priority.squeeze(1) * valid_mask

            if iteration <= cfg.procedure.subdivide_all_until:
                thres = -1
            else:
                thres = priority.quantile(1 - cfg.procedure.subdivide_prop)

            subdivide_mask = (priority > thres) & valid_mask

            # In case the number of voxels over the threshold
            max_n_subdiv = round((cfg.procedure.subdivide_max_num - voxel_model.num_voxels) / 7)
            if subdivide_mask.sum() > max_n_subdiv:
                n_removed = subdivide_mask.sum() - max_n_subdiv
                subdivide_mask &= (priority > priority[subdivide_mask].sort().values[n_removed-1])

            # Subdivision
            voxel_model.subdividing(subdivide_mask)

            # Show statistic
            new_n = voxel_model.num_voxels
            in_p = voxel_model.inside_mask.float().mean().item()
            print(f'[SUBDIVIDING] {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f}; inside={in_p*100:.1f}%)')

            # Reset priority for the next round
            voxel_model.reset_subdivision_priority()

        if need_pruning or need_subdividing:
            # Re-create trainer for the updated parameters
            optimizer, scheduler = create_trainer()
            scheduler.load_state_dict(scheduler_state)
            del scheduler_state

            torch.cuda.empty_cache()

        ######################################################
        # End of adaptive voxels procedure
        ######################################################

        # Update learning rate
        scheduler.step()

        # End processing time tracking of this iteration
        iter_end.record()
        torch.cuda.synchronize()
        elapsed += iter_start.elapsed_time(iter_end)

        # Logging
        with torch.no_grad():
            # Metric
            loss = loss.item()
            psnr = -10 * np.log10(mse.item())

            # Progress bar
            ema_p = max(0.01, 1 / (iteration - first_iter + 1))
            ema_loss_for_log += ema_p * (loss - ema_loss_for_log)
            ema_psnr_for_log += ema_p * (psnr - ema_psnr_for_log)
            if iteration % 10 == 0:
                pb_text = {
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "psnr": f"{ema_psnr_for_log:.2f}",
                }
                progress_bar.set_postfix(pb_text)
                progress_bar.update(10)
            if iteration == cfg.procedure.n_iter:
                progress_bar.close()

            # Log and save
            training_report(
                args=args,
                data_pack=data_pack,
                voxel_model=voxel_model,
                iteration=iteration,
                elapsed=elapsed,
                ema_psnr=ema_psnr_for_log)

            timer.report(iteration)

def training_report(args, data_pack, voxel_model, iteration, elapsed, ema_psnr):

    voxel_model.freeze_vox_geo()

    # Progress view
    if args.pg_view_every > 0 and (iteration % args.pg_view_every == 0 or iteration == 1):
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        if len(test_cameras) == 0:
            test_cameras = data_pack.get_train_cameras()
        pg_idx = 0
        view = test_cameras[pg_idx]
        render_pkg = voxel_model.render(view, output_depth=True, output_normal=True, output_T=True)
        render_image = render_pkg['color']
        render_depth = render_pkg['depth'][0]
        render_depth_med = render_pkg['depth'][2]
        render_normal = render_pkg['normal']
        render_alpha = 1 - render_pkg['T'][0]

        im = np.concatenate([
            np.concatenate([
                im_tensor2np(render_image),
                im_tensor2np(render_alpha)[...,None].repeat(3, axis=-1),
            ], axis=1),
            np.concatenate([
                viz_tensordepth(render_depth, render_alpha),
                im_tensor2np(render_normal * 0.5 + 0.5),
            ], axis=1),
            np.concatenate([
                im_tensor2np(view.depth2normal(render_depth) * 0.5 + 0.5),
                im_tensor2np(view.depth2normal(render_depth_med) * 0.5 + 0.5),
            ], axis=1),
        ], axis=0)
        torch.cuda.empty_cache()

        outdir = os.path.join(args.model_path, "pg_view")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.jpg")
        os.makedirs(outdir, exist_ok=True)

        imageio.imwrite(outpath, im)

        eps_file = os.path.join(args.model_path, "pg_view", "eps.txt")
        with open(eps_file, 'a') as f:
            f.write(f"{iteration},{elapsed/1000:.1f}\n")

    # Report test and samples of training set
    if iteration in args.test_iterations:
        print(f"[EVAL] running...")
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        save_every = max(1, len(test_cameras) // 8)
        outdir = os.path.join(args.model_path, "test_view")
        os.makedirs(outdir, exist_ok=True)
        psnr_lst = []
        video = []
        max_w = torch.zeros([voxel_model.num_voxels, 1], dtype=torch.float32, device="cuda")
        for idx, camera in enumerate(test_cameras):
            render_pkg = voxel_model.render(camera, output_normal=True, track_max_w=True)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            video.append(im)
            if idx % save_every == 0:
                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}.jpg")
                cat = np.concatenate([gt, im], axis=1)
                imageio.imwrite(outpath, cat)

                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}_normal.jpg")
                render_normal = render_pkg['normal']
                render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
                imageio.imwrite(outpath, render_normal)
            mse = np.square(im/255 - gt/255).mean()
            psnr_lst.append(-10 * np.log10(mse))
            max_w = torch.maximum(max_w, render_pkg['max_w'])
        avg_psnr = np.mean(psnr_lst)
        imageio.mimwrite(
            os.path.join(outdir, f"video_iter{iteration:06d}.mp4"),
            video, fps=30)
        torch.cuda.empty_cache()

        fps = time.time()
        for idx, camera in enumerate(test_cameras):
            voxel_model.render(camera, track_max_w=False)
        torch.cuda.synchronize()
        fps = len(test_cameras) / (time.time() - fps)
        torch.cuda.empty_cache()

        # Sample training views to render
        train_cameras = data_pack.get_train_cameras()
        for idx in range(0, len(train_cameras), max(1, len(train_cameras)//8)):
            camera = train_cameras[idx]
            render_pkg = voxel_model.render(
                camera, output_normal=True, track_max_w=True,
                use_auto_exposure=cfg.auto_exposure.enable)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}.jpg")
            cat = np.concatenate([gt, im], axis=1)
            imageio.imwrite(outpath, cat)

            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}_normal.jpg")
            render_normal = render_pkg['normal']
            render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
            imageio.imwrite(outpath, render_normal)

        print(f"[EVAL] iter={iteration:6d}  psnr={avg_psnr:.2f}  fps={fps:.0f}")

        outdir = os.path.join(args.model_path, "test_stat")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.json")
        os.makedirs(outdir, exist_ok=True)
        with open(outpath, 'w') as f:
            q = torch.linspace(0,1,5, device="cuda")
            max_w_q = max_w.quantile(q).tolist()
            peak_mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 ** 3
            stat = {
                'psnr': avg_psnr,
                'ema_psnr': ema_psnr,
                'elapsed': elapsed,
                'fps': fps,
                'n_voxels': voxel_model.num_voxels,
                'max_w_q': max_w_q,
                'peak_mem': peak_mem,
            }
            json.dump(stat, f, indent=4)

    voxel_model.unfreeze_vox_geo()

'''IN PROGRESS - Getting Nerfbuster diffusion model setup'''
def load_diffusion_model(diffusion_config_path, diffusion_ckpt_path, device):
    import yaml
    from dotmap import DotMap
    from nerfbusters.lightning.nerfbusters_trainer import NerfbustersTrainer

    config = yaml.load(open(diffusion_config_path, "r"), Loader=yaml.Loader)
    config = DotMap(config)
    model = NerfbustersTrainer(config)
    ckpt = torch.load(diffusion_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model = model.to(device)
    model.noise_scheduler.alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(device)
    model.dsds_loss.alphas = model.dsds_loss.alphas.to(device)
    print("Loaded diffusion config from", diffusion_config_path)
    print("Loaded diffusion checkpoint from", diffusion_ckpt_path)
    return model

def octree_to_regular_cubes(voxel_model, cube_size=32, num_cubes=4, cube_world_size=6.0):
    """
    Convert octree voxels to regular cube grids for diffusion model
    """
    density_field = voxel_model._geo_grid_pts  # [N, 1] 
    vox_centers = voxel_model._vox_center       # [M, 3] coordinates
    vox_sizes = voxel_model._vox_size          # [M, 1] sizes
    
    # Trim to match sizes (in case of mismatch)
    min_len = min(len(density_field), len(vox_centers))
    density_field = density_field[:min_len]
    vox_centers = vox_centers[:min_len]  
    vox_sizes = vox_sizes[:min_len]
    
    print(f"Using {min_len} voxels for cube generation")
    
    # IMPROVEMENT: Find regions with higher density (actual geometry)
    # Filter for voxels with density above threshold
    density_threshold = density_field.mean() + 0.5 * density_field.std()  # Above average density
    valid_mask = (density_field.squeeze() > density_threshold)
    
    if valid_mask.sum() > 100:  # If we have enough valid voxels
        valid_centers = vox_centers[valid_mask]
        valid_densities = density_field[valid_mask]
        print(f"Found {valid_mask.sum()} voxels with density > {density_threshold:.3f}")
    else:
        # Fall back to all voxels if not enough valid ones
        valid_centers = vox_centers
        valid_densities = density_field
        print("Using all voxels (not enough high-density voxels found)")
    
    cubes = []
    
    for i in range(num_cubes):
        # Sample cube center near existing geometry
        if len(valid_centers) > 0:
            # Pick a random high-density voxel as cube center
            rand_idx = torch.randint(0, len(valid_centers), (1,)).item()
            cube_center = valid_centers[rand_idx]
        else:
            # Fallback to scene center
            scene_min = vox_centers.min(dim=0)[0]
            scene_max = vox_centers.max(dim=0)[0]
            cube_center = (scene_min + scene_max) / 2
        
        # Define cube bounds
        half_size = cube_world_size / 2
        cube_min = cube_center - half_size
        cube_max = cube_center + half_size
        
        print(f"Cube {i}: center={cube_center.cpu().numpy()}, bounds=[{cube_min[0]:.2f}, {cube_max[0]:.2f}]")
        
        # Create regular grid within this cube
        x = torch.linspace(cube_min[0], cube_max[0], cube_size, device=vox_centers.device)
        y = torch.linspace(cube_min[1], cube_max[1], cube_size, device=vox_centers.device)  
        z = torch.linspace(cube_min[2], cube_max[2], cube_size, device=vox_centers.device)
        
        # Create 3D meshgrid
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)  # [32^3, 3]
        
        # Find nearest octree voxels for each grid point
        cube_densities = torch.zeros(cube_size**3, device=density_field.device)
        
        for j, grid_point in enumerate(grid_points):
            # Find closest voxel center
            distances = torch.norm(vox_centers - grid_point.unsqueeze(0), dim=1)
            closest_idx = torch.argmin(distances)
            
            # Use that voxel's density
            cube_densities[j] = density_field[closest_idx, 0]
        
        # Reshape to 3D cube
        cube = cube_densities.reshape(cube_size, cube_size, cube_size)
        cubes.append(cube)
        
        # Show some stats
        print(f"  Cube {i} density range: [{cube.min():.3f}, {cube.max():.3f}]")
    
    # Stack into batch format
    density_cubes = torch.stack(cubes, dim=0)  # [num_cubes, 32, 32, 32]
    density_cubes = density_cubes.unsqueeze(1)  # [num_cubes, 1, 32, 32, 32]
    
    # Convert from log-density to occupancy for diffusion model
    # Clamp extreme values first
    density_cubes = torch.clamp(density_cubes, min=-15, max=5)
    occupancy = torch.sigmoid(density_cubes)  # Convert log-density to probability
    normalized_cubes = occupancy * 2 - 1      # Scale to [-1, 1]
    
    print(f"\nFinal cubes shape: {normalized_cubes.shape}")
    print(f"Final cubes range: [{normalized_cubes.min():.3f}, {normalized_cubes.max():.3f}]")
    
    return normalized_cubes

# ==================== NERFBUSTERS INTEGRATION FUNCTIONS ====================

def sample_cubes(min_x, min_y, min_z, max_x, max_y, max_z, res, spr_min, spr_max, num_cubes, 
                cube_centers_x=None, cube_centers_y=None, cube_centers_z=None, device=None):
    """Nerfbusters cube sampling function"""
    assert device is not None, "device must be specified"
    
    scales = torch.rand(num_cubes, device=device) * (spr_max - spr_min) + spr_min
    cube_len = (max_x - min_x) * scales
    half_cube_len = cube_len / 2
    
    if cube_centers_x is None:
        cube_centers_x = (torch.rand(num_cubes, device=device) * (max_x - min_x - 2.0 * half_cube_len) + min_x + half_cube_len)
        cube_centers_y = (torch.rand(num_cubes, device=device) * (max_y - min_y - 2.0 * half_cube_len) + min_y + half_cube_len)
        cube_centers_z = (torch.rand(num_cubes, device=device) * (max_z - min_z - 2.0 * half_cube_len) + min_z + half_cube_len)
    else:
        cube_centers_x = cube_centers_x * (max_x - min_x - 2.0 * half_cube_len) + min_x + half_cube_len
        cube_centers_y = cube_centers_y * (max_y - min_y - 2.0 * half_cube_len) + min_y + half_cube_len
        cube_centers_z = cube_centers_z * (max_z - min_z - 2.0 * half_cube_len) + min_z + half_cube_len
    
    cube_start_x = cube_centers_x - half_cube_len
    cube_start_y = cube_centers_y - half_cube_len
    cube_start_z = cube_centers_z - half_cube_len
    cube_start_xyz = torch.stack([cube_start_x, cube_start_y, cube_start_z], dim=-1).reshape(num_cubes, 1, 1, 1, 3)
    
    cube_end_x = cube_centers_x + half_cube_len
    cube_end_y = cube_centers_y + half_cube_len
    cube_end_z = cube_centers_z + half_cube_len
    cube_end_xyz = torch.stack([cube_end_x, cube_end_y, cube_end_z], dim=-1).reshape(num_cubes, 1, 1, 1, 3)
    
    l = torch.linspace(0, 1, res, device=device)
    xyz = torch.stack(torch.meshgrid([l, l, l], indexing="ij"), dim=-1)
    xyz = xyz[None, ...] * (cube_end_xyz - cube_start_xyz) + cube_start_xyz
    
    return xyz, cube_start_xyz, cube_end_xyz, scales

class SVRWeightGrid(torch.nn.Module):
    """Importance sampling grid for SVR"""
    def __init__(self, resolution=64, device="cuda"):
        super().__init__()
        self.resolution = resolution
        self.register_buffer("_weights", torch.ones(resolution, resolution, resolution, device=device))
        self.scene_bounds = None
        
    def update_from_svr(self, voxel_model, ema_decay=0.9):
        """Update grid from SVR voxel centers and densities"""
        vox_centers = voxel_model._vox_center
        densities = voxel_model._geo_grid_pts.squeeze()
        
        # Get/store scene bounds
        if self.scene_bounds is None:
            min_bounds = vox_centers.min(dim=0)[0] - 3.0
            max_bounds = vox_centers.max(dim=0)[0] + 3.0
            self.scene_bounds = (min_bounds, max_bounds)
        else:
            min_bounds, max_bounds = self.scene_bounds
        
        # Normalize positions to [0, 1]
        scene_size = max_bounds - min_bounds + 1e-8
        normalized_pos = (vox_centers - min_bounds) / scene_size
        normalized_pos = torch.clamp(normalized_pos, 0.0, 0.999)
        
        # Convert to grid indices
        indices = (normalized_pos * self.resolution).long()
        
        # Apply EMA decay
        self._weights = ema_decay * self._weights
        
        # Convert log-densities to positive weights
        positive_densities = torch.clamp(torch.exp(densities + 10), 0.1, 10.0)  # Add offset for log-densities
        
        # Accumulate weights efficiently
        flat_indices = indices[:, 0] * self.resolution**2 + indices[:, 1] * self.resolution + indices[:, 2]
        self._weights.view(-1).scatter_add_(0, flat_indices, positive_densities)
    
    def sample_importance_centers(self, num_cubes):
        """Sample cube centers from high-importance regions"""
        device = self._weights.device
        
        # Create probability distribution
        probs = self._weights.view(-1) / (self._weights.view(-1).sum() + 1e-8)
        
        # Sample indices
        dist = torch.distributions.categorical.Categorical(probs)
        sample_indices = dist.sample((num_cubes,))
        
        # Convert to 3D coordinates
        res = self.resolution
        z = sample_indices % res
        y = (sample_indices // res) % res
        x = sample_indices // (res * res)
        
        # Convert to normalized coordinates with jitter
        coords = torch.stack([x, y, z], dim=1).float()
        coords = (coords + torch.rand_like(coords)) / res
        
        return coords

def query_svr_density(voxel_model, xyz_points, k_neighbors=8):
    """
    Query SVR density at arbitrary 3D coordinates using differentiable interpolation
    """
    vox_centers = voxel_model._vox_center
    density_field = voxel_model._geo_grid_pts.squeeze()
    
    xyz_points = xyz_points.reshape(-1, 3)
    batch_size = xyz_points.shape[0]
    
    # Efficient batch processing
    chunk_size = 16384  # Process in chunks to avoid memory issues
    all_densities = []
    
    for i in range(0, batch_size, chunk_size):
        chunk_points = xyz_points[i:i+chunk_size]
        chunk_densities = torch.zeros(chunk_points.shape[0], device=xyz_points.device)
        
        for j, query_point in enumerate(chunk_points):
            # Compute distances to all voxels
            distances = torch.norm(vox_centers - query_point.unsqueeze(0), dim=1)
            
            # Find k nearest neighbors
            _, top_k_indices = torch.topk(distances, k_neighbors, largest=False)
            top_k_distances = distances[top_k_indices]
            top_k_densities = density_field[top_k_indices]
            
            # Differentiable interpolation using softmax weights
            weights = torch.softmax(-top_k_distances / 0.5, dim=0)  # Temperature = 0.5
            chunk_densities[j] = torch.sum(weights * top_k_densities)
        
        all_densities.append(chunk_densities)
    
    return torch.cat(all_densities, dim=0)

def density_to_x_svr(density):
    """Convert SVR density to diffusion model input range [-1, 1] with consistent dtype"""
    # For log-densities around -10 to -9, normalize appropriately
    normalized = torch.sigmoid(density + 10)  # Shift and sigmoid
    result = normalized * 2 - 1  # Scale to [-1, 1]
    
    # 🔧 FIX: Ensure float32 output
    return result.float()

def nerfbusters_importance_sampling(voxel_model, diffusion_model, weight_grid, iteration, num_cubes=20):
    """
    Complete Nerfbusters integration with importance sampling
    """
    # Update weight grid periodically
    if iteration % 100 == 0:
        with timer.time("TIMER - Updating Weight Grid"):
            weight_grid.update_from_svr(voxel_model)
    
    with timer.time("TIMER - Grabbing Scene Bounds"): # Get scene bounds from weight grid
        if weight_grid.scene_bounds is None:
            return torch.tensor(0.0, device=voxel_model._geo_grid_pts.device)
    
    min_bounds, max_bounds = weight_grid.scene_bounds
    
    # Sample cube centers using importance sampling (70%) + random (30%)
    num_importance = int(0.7 * num_cubes)
    num_random = num_cubes - num_importance
    
    with timer.time("TIMER - Sampling Importance Centers"):
        importance_centers = weight_grid.sample_importance_centers(num_importance)
    random_centers = torch.rand(num_random, 3, device=importance_centers.device)
    all_centers = torch.cat([importance_centers, random_centers], dim=0)
    
    # Convert normalized centers to world coordinates
    world_centers = all_centers * (max_bounds - min_bounds) + min_bounds
    
    # Sample cubes using Nerfbusters method
    with timer.time("TIMER - Sampling the Cubes"):
        xyz, _, _, scales = sample_cubes(
            min_bounds[0], min_bounds[1], min_bounds[2],
            max_bounds[0], max_bounds[1], max_bounds[2],
            res=16, #THIS WAS 32, but changed to 16 beacuse of long training times
            spr_min=0.02, spr_max=0.08,  # 2-8% of scene
            num_cubes=num_cubes,
            cube_centers_x=world_centers[:, 0],
            cube_centers_y=world_centers[:, 1],
            cube_centers_z=world_centers[:, 2],
            device=voxel_model._geo_grid_pts.device
        )
    
    # Query SVR density at cube points (DIFFERENTIABLE!)
    xyz_flat = xyz.reshape(-1, 3)
    with timer.time("TIMER - Query SVR Density"):
        densities_flat = query_svr_density(voxel_model, xyz_flat)
    density_cubes = densities_flat.reshape(num_cubes, 32, 32, 32)
    
    # Convert to diffusion model format
    with timer.time("TIMER - Converting to diff format"):
        x = density_to_x_svr(density_cubes).unsqueeze(1)  # Add channel dimension
    
    # Apply diffusion model
    unet_model = diffusion_model.model
    timestep = torch.tensor([150], device=x.device)  # Use timestep 15
    
    with timer.time("TIMER - Running through unet"):
        diffusion_output = unet_model(x, timestep)
    enhanced_x = diffusion_output.sample
    
    # Compute DSDS-style loss
    diff_loss = torch.nn.functional.mse_loss(enhanced_x, x)
    
    # Optional: Add sparsity regularization
    sparsity_loss = ((x + 1.0) ** 2).mean() * 0.01
    
    total_loss = diff_loss + sparsity_loss
    
    return total_loss, xyz, scales

#TEST NERFBUSTERS INTEGRATION
def test_nerfbusters_integration(voxel_model, diffusion_model, weight_grid):
    print("🧪 Testing Nerfbusters integration...")
    try:
        # Test weight grid update
        '''
        weight_grid.update_from_svr(voxel_model)
        print("✅ Weight grid update: OK")
        
        # Test importance sampling
        centers = weight_grid.sample_importance_centers(5)
        print(f"✅ Importance sampling: {centers.shape}")
        '''
        # Test density querying
        test_points = torch.randn(100, 3).cuda() * 10  # Random test points
        densities = query_svr_density(voxel_model, test_points)
        print(f"✅ Density querying: {densities.shape}, range [{densities.min():.3f}, {densities.max():.3f}]")
        
        # Test full pipeline #ADD BACK WEIGHT GRIED IF USING
        loss, xyz, scales = nerfbusters_random_sampling(voxel_model, diffusion_model, 1, num_cubes=3)
        print(f"✅ Full pipeline: Loss={loss.item():.6f}, Cubes={xyz.shape}")
        
        print("🎉 ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

#THRESHOLDING VOXELS METHODS
def extract_view_relevant_cubes(voxel_model, cam, cube_size=16, num_cubes=8):
    """
    Extract density cubes from regions visible in current camera view  
    """
    device = voxel_model._geo_grid_pts.device
    
    # Get scene bounds from voxel model
    vox_centers = voxel_model._vox_center
    scene_min = vox_centers.min(dim=0)[0] - 2.0
    scene_max = vox_centers.max(dim=0)[0] + 2.0
    
    # Sample cubes preferentially along camera view direction
    cam_pos = cam.camera_center if hasattr(cam, 'camera_center') else torch.zeros(3, device=device)
    cam_forward = cam.get_world_directions()[cam.image.shape[0]//2, cam.image.shape[1]//2] if hasattr(cam, 'get_world_directions') else torch.tensor([0,0,1], device=device)
    
    # Sample cubes along viewing direction
    centers = []
    for i in range(num_cubes):
        # Sample at different depths along camera ray
        depth = 2.0 + i * 3.0  # Depths from 2 to 20
        cube_center = cam_pos + depth * cam_forward
        
        # Add some randomization
        cube_center += torch.randn(3, device=device) * 1.0
        centers.append(cube_center)
    
    centers = torch.stack(centers)
    
    # Extract cubes at these centers
    xyz, _, _, scales = sample_cubes(
        scene_min[0], scene_min[1], scene_min[2],
        scene_max[0], scene_max[1], scene_max[2],
        res=cube_size,
        spr_min=0.02, spr_max=0.08,
        num_cubes=num_cubes,
        cube_centers_x=centers[:, 0],
        cube_centers_y=centers[:, 1], 
        cube_centers_z=centers[:, 2],
        device=device
    )
    
    # Query densities at cube points
    xyz_flat = xyz.reshape(-1, 3)
    densities_flat = query_svr_density(voxel_model, xyz_flat)
    density_cubes = densities_flat.reshape(num_cubes, cube_size, cube_size, cube_size)
    
    # Convert to diffusion input format
    x = density_to_x_svr(density_cubes).unsqueeze(1)  # [num_cubes, 1, size, size, size]
    
    # 🔧 FIX: Ensure everything is float32
    x = x.float()
    xyz = xyz.float() 
    scales = scales.float()
    
    return x, xyz, scales

def compute_cube_differences(original_cubes, enhanced_cubes, threshold=0.05):
    """
    Compute per-cube difference and confidence scores
    """
    # Compute absolute difference per cube
    diff_per_cube = torch.abs(enhanced_cubes - original_cubes)
    
    # Mean difference per cube
    cube_diff_scores = diff_per_cube.mean(dim=[1,2,3,4])  # [num_cubes]
    
    # Confidence: 1.0 = high confidence (small diff), 0.0 = low confidence (large diff)
    confidence_scores = torch.exp(-cube_diff_scores / threshold)
    
    # Binary mask for high-difference cubes
    high_diff_mask = cube_diff_scores > threshold
    
    return cube_diff_scores, confidence_scores, high_diff_mask

def confidence_weighted_loss(original_cubes, enhanced_cubes, confidence_scores, high_diff_mask):
    """
    Compute loss only on low-confidence (high-difference) regions
    """
    if high_diff_mask.sum() == 0:
        # All cubes are confident, no diffusion needed
        return torch.tensor(0.0, device=original_cubes.device, requires_grad=True)
    
    # Extract uncertain cubes
    uncertain_original = original_cubes[high_diff_mask]
    uncertain_enhanced = enhanced_cubes[high_diff_mask]
    uncertain_confidence = confidence_scores[high_diff_mask]
    
    # Weight loss by confidence (lower confidence = higher weight)
    weights = (1.0 - uncertain_confidence).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
    
    # Weighted MSE loss
    weighted_diff = weights * (uncertain_enhanced - uncertain_original) ** 2
    loss = weighted_diff.mean()
    
    return loss

def alpha_thresholding_diffusion(voxel_model, diffusion_model, cam, iteration, 
                               threshold=0.05, cube_size=16, num_cubes=6):
    """
    Main alpha thresholding diffusion function
    """
    device = voxel_model._geo_grid_pts.device
    
    try:
        # Step 1: Extract view-relevant cubes
        original_cubes, xyz, scales = extract_view_relevant_cubes(
            voxel_model, cam, cube_size=cube_size, num_cubes=num_cubes
        )
        
        original_cubes = original_cubes.float() 

        # Step 2: Apply diffusion model to get enhanced cubes
        unet_model = diffusion_model.model
        timestep = torch.tensor([10], device=device)
        
        # Ensure consistent dtype - no mixed precision for now
        original_cubes = original_cubes.float()
        timestep = timestep.long()  # Timesteps should be integers

        diffusion_output = unet_model(original_cubes, timestep)
        enhanced_cubes = diffusion_output.sample.float()
        
        # Step 3: Compute differences and confidence scores
        cube_diff_scores, confidence_scores, high_diff_mask = compute_cube_differences(
            original_cubes, enhanced_cubes, threshold=threshold
        )
        
        # Step 4: Compute confidence-weighted loss
        alpha_loss = confidence_weighted_loss(
            original_cubes, enhanced_cubes, confidence_scores, high_diff_mask
        )
        
        # Step 5: Add sparsity regularization for uncertain regions
        if high_diff_mask.sum() > 0:
            uncertain_original = original_cubes[high_diff_mask]
            sparsity_loss = ((uncertain_original + 1.0) ** 2).mean() * 0.01
            alpha_loss = alpha_loss + sparsity_loss
        
        # Debug info
        if iteration % 100 == 0:
            print(f"[ALPHA-THRESH] Uncertain cubes: {high_diff_mask.sum()}/{num_cubes}")
            print(f"[ALPHA-THRESH] Avg confidence: {confidence_scores.mean():.3f}")
            print(f"[ALPHA-THRESH] Loss: {alpha_loss.item():.6f}")
        
        return alpha_loss, cube_diff_scores.mean()
        
    except Exception as e:
        print(f"Alpha thresholding failed: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0

def adaptive_threshold_schedule(iteration, current_psnr, base_threshold=0.05):
    """
    Adjust threshold based on training progress
    """
    # Early training: Higher threshold (more accepting)
    if iteration < 3000:
        threshold = base_threshold * 2.0
    elif iteration < 10000:
        threshold = base_threshold * 1.5
    else:
        threshold = base_threshold
    
    # Adjust based on current PSNR
    if current_psnr < 24.0:
        # Poor quality, be more accepting of changes
        threshold *= 1.5
    elif current_psnr > 26.0:
        # Good quality, be very selective
        threshold *= 0.3
    
    return threshold

def save_checkpoint(voxel_model, optimizer, scheduler, iteration, model_path, ema_loss, ema_psnr):
    """Save training checkpoint"""
    try:
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model state
        voxel_model.save_iteration(model_path, iteration, quantize=False)
        
        # Save optimizer and scheduler state
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt")
        torch.save({
            'iteration': iteration,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ema_loss': ema_loss,  # Now using the parameter
            'ema_psnr': ema_psnr,  # Now using the parameter
        }, checkpoint_path)
        
        print(f"[CHECKPOINT] Saved at iteration {iteration}: {checkpoint_path}")
        
        # Keep only last 3 checkpoints to save disk space
        cleanup_old_checkpoints(checkpoint_dir, keep_last=3)
        
    except Exception as e:
        print(f"[CHECKPOINT] Failed to save: {e}")

def cleanup_old_checkpoints(checkpoint_dir, keep_last=2):
    """Keep only the most recent checkpoints"""
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")])
    if len(checkpoints) > keep_last:
        for old_checkpoint in checkpoints[:-keep_last]:
            os.remove(os.path.join(checkpoint_dir, old_checkpoint))

def load_checkpoint(voxel_model, optimizer, scheduler, model_path, iteration=None):
    """Load the most recent checkpoint"""
    checkpoint_dir = os.path.join(model_path, "checkpoints")
    
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        return None
    
    if iteration is None:
        # Find most recent checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
        if not checkpoints:
            return None
        
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
        iteration = int(latest_checkpoint.split('_')[1].split('.')[0])
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        # Load voxel model
        loaded_iter = voxel_model.load_iteration(model_path, iteration)
        
        # Load optimizer and scheduler
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"[CHECKPOINT] Loaded from iteration {iteration}")
        return {
            'iteration': checkpoint['iteration'],
            'ema_loss': checkpoint.get('ema_loss', 0.0),
            'ema_psnr': checkpoint.get('ema_psnr', 0.0)
        }
    
    return None

def apply_diffusion_to_density_field(voxel_model, diffusion_model, cam, iteration, 
                                   threshold=0.05, cube_size=16, num_cubes=6, learning_rate=0.01):
    """
    Apply diffusion model corrections directly to the density field
    """
    device = voxel_model._geo_grid_pts.device
    
    try:
        # Extract view-relevant cubes with gradient tracking
        original_cubes, xyz, scales = extract_view_relevant_cubes(
            voxel_model, cam, cube_size=cube_size, num_cubes=num_cubes
        )
        
        # Apply diffusion model
        unet_model = diffusion_model.model
        timestep = torch.tensor([10], device=device, dtype=torch.long)
        
        with torch.no_grad():  # Don't backprop through diffusion model
            diffusion_output = unet_model(original_cubes.float(), timestep)
            enhanced_cubes = diffusion_output.sample.float()
        
        # Compute differences
        cube_diff_scores, confidence_scores, high_diff_mask = compute_cube_differences(
            original_cubes, enhanced_cubes, threshold=threshold
        )
        
        if high_diff_mask.sum() > 0:
            # Apply corrections to uncertain regions
            uncertain_xyz = xyz[high_diff_mask]  # [uncertain_cubes, res, res, res, 3]
            uncertain_original = original_cubes[high_diff_mask]
            uncertain_enhanced = enhanced_cubes[high_diff_mask]
            uncertain_confidence = confidence_scores[high_diff_mask]
            
            # Compute correction strength based on confidence
            correction_strength = (1.0 - uncertain_confidence) * learning_rate
            
            for i, (cube_xyz, orig_cube, enh_cube, strength) in enumerate(
                zip(uncertain_xyz, uncertain_original, uncertain_enhanced, correction_strength)
            ):
                # Convert cube differences back to density field updates
                cube_diff = (enh_cube.squeeze() - orig_cube.squeeze()) * strength
                
                # Apply corrections to nearby voxels in the density field
                cube_points = cube_xyz.reshape(-1, 3)  # [res³, 3]
                cube_corrections = cube_diff.reshape(-1)  # [res³]
                
                # Find nearest voxels for each cube point
                for point, correction in zip(cube_points, cube_corrections):
                    if abs(correction) > 0.001:  # Only apply significant corrections
                        # Find closest voxel
                        distances = torch.norm(voxel_model._vox_center - point, dim=1)
                        closest_idx = torch.argmin(distances)
                        
                        # Apply correction with distance weighting
                        if distances[closest_idx] < 2.0:  # Within reasonable distance
                            weight = torch.exp(-distances[closest_idx] / 0.5)
                            voxel_model._geo_grid_pts[closest_idx] += correction * weight * 0.1
        
        # Return diagnostic info
        return cube_diff_scores.mean(), high_diff_mask.sum()
        
    except Exception as e:
        print(f"Diffusion density integration failed: {e}")
        return 0.0, 0


def alpha_thresholding_diffusion_with_gradients(voxel_model, diffusion_model, cam, iteration, 
                                              threshold=0.05, cube_size=16, num_cubes=6):
    """
    Alpha thresholding diffusion that creates gradients to the density field
    """
    device = voxel_model._geo_grid_pts.device
    
    try:
        # Extract cubes WITH gradient tracking
        original_cubes, xyz, scales = extract_view_relevant_cubes(
            voxel_model, cam, cube_size=cube_size, num_cubes=num_cubes
        )
        
        # Get diffusion target (without gradients)
        with torch.no_grad():
            unet_model = diffusion_model.model
            timestep = torch.tensor([10], device=device, dtype=torch.long)
            diffusion_output = unet_model(original_cubes.float(), timestep)
            enhanced_cubes = diffusion_output.sample.float()
        
        # Compute differences and create loss that flows back to density field
        cube_diff_scores, confidence_scores, high_diff_mask = compute_cube_differences(
            original_cubes, enhanced_cubes, threshold=threshold
        )
        
        if high_diff_mask.sum() > 0:
            # Create loss that will update the density field
            uncertain_original = original_cubes[high_diff_mask]
            uncertain_enhanced = enhanced_cubes[high_diff_mask]
            uncertain_confidence = confidence_scores[high_diff_mask]
            
            # Weight by confidence - focus on uncertain regions
            weights = (1.0 - uncertain_confidence).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
            # MSE loss between current cubes and diffusion target
            # This creates gradients that flow back through query_svr_density to _geo_grid_pts
            weighted_diff = weights * (uncertain_original - uncertain_enhanced.detach()) ** 2
            alpha_loss = weighted_diff.mean()
            
            return alpha_loss, cube_diff_scores.mean()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0
        
    except Exception as e:
        print(f"Alpha thresholding with gradients failed: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0

def nerfbusters_random_sampling(voxel_model, diffusion_model, iteration, num_cubes=20):
    """
    Complete Nerfbusters integration with RANDOM sampling (no importance sampling)
    """
    # Get scene bounds from voxel model directly
    vox_centers = voxel_model._vox_center
    scene_min = vox_centers.min(dim=0)[0] - 3.0
    scene_max = vox_centers.max(dim=0)[0] + 3.0
    
    with timer.time("TIMER - Random Cube Sampling"):
        # Sample cubes using completely random centers
        xyz, _, _, scales = sample_cubes(
            scene_min[0], scene_min[1], scene_min[2],
            scene_max[0], scene_max[1], scene_max[2],
            res=32,
            spr_min=0.02, spr_max=0.08,  # 2-8% of scene
            num_cubes=num_cubes,
            cube_centers_x=None,  # Let it choose random centers
            cube_centers_y=None,
            cube_centers_z=None,
            device=voxel_model._geo_grid_pts.device
        )

    # Query SVR density at cube points (DIFFERENTIABLE!)
    xyz_flat = xyz.reshape(-1, 3)
    with timer.time("TIMER - Query SVR Density Random"):
        densities_flat = query_svr_density(voxel_model, xyz_flat)
    density_cubes = densities_flat.reshape(num_cubes, 32, 32, 32)

    # Convert to diffusion model format
    with timer.time("TIMER - Converting to diff format"):
        x = density_to_x_svr(density_cubes).unsqueeze(1)  # Add channel dimension

    # Apply diffusion model
    unet_model = diffusion_model.model
    timestep = torch.tensor([15], device=x.device)  # Use timestep 15
    with timer.time("TIMER - Running through unet"):
        diffusion_output = unet_model(x, timestep)
    enhanced_x = diffusion_output.sample

    # Compute DSDS-style loss
    diff_loss = torch.nn.functional.mse_loss(enhanced_x, x)
    # Optional: Add sparsity regularization
    sparsity_loss = ((x + 1.0) ** 2).mean() * 0.01
    total_loss = diff_loss + sparsity_loss

    return total_loss, xyz, scales

def apply_thresholding_to_density_field(voxel_model, diffusion_model, iteration, 
                                       threshold=0.1, num_cubes=10, attenuation_factor=0.3):
    """
    Direct density field thresholding - no loss calculation
    Compare original vs diffusion-enhanced density, reduce density in high-difference regions
    """
    device = voxel_model._geo_grid_pts.device
    
    # Sample random cubes from scene
    vox_centers = voxel_model._vox_center
    scene_min = vox_centers.min(dim=0)[0] - 3.0
    scene_max = vox_centers.max(dim=0)[0] + 3.0
    
    with timer.time("TIMER - Random Cube Sampling"):
        xyz, _, _, scales = sample_cubes(
            scene_min[0], scene_min[1], scene_min[2],
            scene_max[0], scene_max[1], scene_max[2],
            res=32,
            spr_min=0.02, spr_max=0.08,
            num_cubes=num_cubes,
            cube_centers_x=None,  # Random sampling
            cube_centers_y=None,
            cube_centers_z=None,
            device=device
        )
    with timer.time("TIMER - Query Original Density"):
        # Query current density field
        xyz_flat = xyz.reshape(-1, 3)
        original_densities_flat = query_svr_density(voxel_model, xyz_flat)
        original_density_cubes = original_densities_flat.reshape(num_cubes, 32, 32, 32)
        print(f"Raw density stats: min={original_density_cubes.min():.6f}, max={original_density_cubes.max():.6f}, mean={original_density_cubes.mean():.6f}")
    with timer.time("TIMER - Generate Enhanced Density"):
        # Convert to diffusion format and enhance
        #x = density_to_x_svr(original_density_cubes).unsqueeze(1)
        x = density_to_x_nerfbusters(original_density_cubes, 
                            crossing=-8, 
                            activation="binarize").unsqueeze(1)
        print(f"DEBUG: x stats - min: {x.min():.4f}, max: {x.max():.4f}, has_nan: {torch.isnan(x).any()}")
        
        # Apply diffusion model
        with torch.no_grad():
            unet_model = diffusion_model.model
            timestep = torch.tensor([15], device=x.device)
            diffusion_output = unet_model(x, timestep)
            enhanced_x = diffusion_output.sample

        print(f"DEBUG: enhanced_x stats - min: {enhanced_x.min():.4f}, max: {enhanced_x.max():.4f}, has_nan: {torch.isnan(enhanced_x).any()}")
        
        # Convert back to density format
        crossing = -8
        
        '''
        #Uniform Clamp
        enhanced_density_cubes = torch.where(enhanced_x.squeeze() < 0, 
                                    crossing - 0.5,  # Empty: -8.5
                                    crossing + 0.5)  # Occupied: -7.5
        '''
        #Continuous Mapping Clamp
        enhanced_density_cubes = torch.clamp(enhanced_x.squeeze() * 0.5 - 8.0, -10.0, -6.0)
        print(f"DEBUG: enhanced_density_cubes stats - min: {enhanced_density_cubes.min():.4f}, max: {enhanced_density_cubes.max():.4f}, has_nan: {torch.isnan(enhanced_density_cubes).any()}")

    with timer.time("TIMER - Compare and Threshold"):
        # Compute differences
        density_differences = torch.abs(enhanced_density_cubes - original_density_cubes)
        print(f"DEBUG: density_differences stats - min: {density_differences.min():.4f}, max: {density_differences.max():.4f}, has_nan: {torch.isnan(density_differences).any()}")
        
        # Create threshold mask
        high_diff_mask = density_differences > threshold
        
        # Statistics
        total_voxels = density_differences.numel()
        affected_voxels = high_diff_mask.sum().item()
        avg_difference = density_differences.mean().item()
        
        if iteration % 100 == 0:
            print(f"[THRESHOLDING] Threshold: {threshold:.3f}")
            print(f"[THRESHOLDING] Affected voxels: {affected_voxels}/{total_voxels} ({100*affected_voxels/total_voxels:.1f}%)")
            print(f"[THRESHOLDING] Avg difference: {avg_difference:.4f}")
    with timer.time("TIMER - Apply Density Reduction"):
        # Apply thresholding - reduce density in high-difference regions directly
        with torch.no_grad():  # No gradients needed for direct modification
            # Get flat indices of high-difference voxels across all cubes
            high_diff_flat_indices = torch.where(high_diff_mask.flatten())[0]
            
            if len(high_diff_flat_indices) > 0:
                # Get corresponding positions in xyz_flat
                high_diff_positions = xyz_flat[high_diff_flat_indices]  # [N_affected, 3]
                
                # Query the current densities at these positions directly
                current_high_diff_densities = query_svr_density(voxel_model, high_diff_positions)
                
                # Apply attenuation directly
                attenuated_densities = current_high_diff_densities * (1 - attenuation_factor)
                
                # Update the density field directly at these positions
                # This assumes query_svr_density and update operations work on the same spatial mapping
                for idx, pos in enumerate(high_diff_positions):
                    # Find the voxel index that corresponds to this position
                    # This is more efficient than computing all distances
                    distances = torch.norm(vox_centers - pos, dim=1)
                    closest_voxel_idx = torch.argmin(distances)
                    
                    # Only update if reasonably close (spatial consistency check)
                    if distances[closest_voxel_idx] < 1.0:
                        voxel_model._geo_grid_pts[closest_voxel_idx, 0] = attenuated_densities[idx]
                        
        if iteration % 100 == 0:
            print(f"[THRESHOLDING] Applied {attenuation_factor:.1f} attenuation to noisy regions")

def adaptive_threshold_schedule(iteration, current_psnr):
    """
    Adaptive threshold based on training progress
    """
    base_threshold = 0.1
    
    # Early training: More lenient threshold
    if iteration < 2000:
        threshold = base_threshold * 2.0
    elif iteration < 5000:
        threshold = base_threshold * 1.5
    else:
        threshold = base_threshold
    
    # Adjust based on PSNR quality
    if current_psnr < 20.0:
        threshold *= 1.5  # Be more accepting of changes
    elif current_psnr > 25.0:
        threshold *= 0.7  # Be more selective
    
    return threshold


def density_to_x_nerfbusters(density, crossing=0.01, activation="binarize", max_val=500.0):
    """Full Nerfbusters density_to_x conversion"""
    if activation == "binarize":
        x = torch.where(density.detach() < crossing, -1.0, 1.0)
    elif activation == "sigmoid":
        x = 2 * torch.sigmoid(1000.0 * (density - crossing)) - 1.0
    elif activation == "clamp":
        x = torch.clamp(crossing * density - 1, -1, 1)
    elif activation == "rescale_clamp":
        x = torch.clamp(density / crossing - 1.0, -1.0, 1.0)
    # Add other cases if needed
    else:
        raise ValueError(f"Unknown activation: {activation}")
    return x

if __name__ == "__main__":

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster optimization."
        "You can specify a list of config files to overwrite the default setups."
        "All config fields can also be overwritten by command line.")
    parser.add_argument('--model_path')
    parser.add_argument('--cfg_files', default=[], nargs='*')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[-1])
    parser.add_argument("--pg_view_every", type=int, default=200)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--load_optimizer", action='store_true')
    parser.add_argument("--save_optimizer", action='store_true')
    parser.add_argument("--save_quantized", action='store_true')
    args, cmd_lst = parser.parse_known_args()

    # Update config from files and command line
    update_config(args.cfg_files, cmd_lst)

    # Global init
    seed_everything(cfg.procedure.seed)
    torch.cuda.set_device(torch.device("cuda:0"))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Setup output folder and dump config
    if not args.model_path:
        datetime_str = datetime.datetime.now().strftime("%Y-%m%d-%H%M")
        unique_str = str(uuid.uuid4())[:6]
        folder_name = f"{datetime_str}-{unique_str}"
        args.model_path = os.path.join(f"./output", folder_name)

    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "config.yaml"), "w") as f:
            f.write(cfg.dump())
    print(f"Output folder: {args.model_path}")

    # Apply scheduler scaling
    if cfg.procedure.sche_mult != 1:
        sche_mult = cfg.procedure.sche_mult

        for key in ['geo_lr', 'sh0_lr', 'shs_lr']:
            cfg.optimizer[key] /= sche_mult
        cfg.optimizer.lr_decay_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in cfg.optimizer.lr_decay_ckpt]

        for key in [
                'dist_from', 'tv_from', 'tv_until',
                'n_dmean_from', 'n_dmean_end',
                'n_dmed_from', 'n_dmed_end',
                'depthanythingv2_from', 'depthanythingv2_end',
                'mast3r_metric_depth_from', 'mast3r_metric_depth_end']:
            cfg.regularizer[key] = round(cfg.regularizer[key] * sche_mult)

        for key in [
                'n_iter',
                'adapt_from', 'adapt_every',
                'prune_until', 'subdivide_until', 'subdivide_all_until']:
            cfg.procedure[key] = round(cfg.procedure[key] * sche_mult)
        cfg.procedure.reset_sh_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in cfg.procedure.reset_sh_ckpt]

    # Update negative iterations
    for i in range(len(args.test_iterations)):
        if args.test_iterations[i] < 0:
            args.test_iterations[i] += cfg.procedure.n_iter + 1
    for i in range(len(args.checkpoint_iterations)):
        if args.checkpoint_iterations[i] < 0:
            args.checkpoint_iterations[i] += cfg.procedure.n_iter + 1

    # Launch training loop
    training(args)
    print("Everything done.")
