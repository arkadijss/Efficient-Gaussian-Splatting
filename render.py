#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss, ssim
import time
import copy
import numpy as np
import json
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp,overwrite, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    density_path = os.path.join(model_path, name, "ours_{}".format(iteration), "density")
    opacity_path = os.path.join(model_path, name, "ours_{}".format(iteration), "opacity")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    if os.path.exists(os.path.join(model_path, name, "ours_{}".format(iteration),"stats.json")) and not overwrite:
        return

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(density_path, exist_ok=True)
    makedirs(opacity_path, exist_ok=True)

    stats = {"l1_loss" : [], "psnr" : [],"ssim" : [],"lpips" : [],"fps" : [], "num_gaussians" : gaussians.get_xyz.shape[0]}

    mask = lambda l : l
    density_gaussians = copy.deepcopy(gaussians)
    density_gaussians._xyz = mask(gaussians._xyz)
    density_gaussians._rotation = mask(gaussians._rotation)
    density_gaussians._opacity = mask(-torch.ones_like(gaussians._opacity)*1)
    density_gaussians._scaling = mask(-torch.ones_like(gaussians._scaling)*10)
    density_gaussians._features_dc = mask(torch.ones_like(gaussians._features_dc))
    density_gaussians._features_rest = mask(gaussians._features_rest)

    #mask = lambda l : l[gaussians.get_opacity[:,0] > 0.9]
    opacity_gaussians = copy.deepcopy(gaussians)
    opacity_gaussians._xyz = mask(gaussians._xyz)
    opacity_gaussians._rotation = mask(gaussians._rotation)
    opacity_gaussians._opacity = mask(gaussians._opacity/(gaussians._opacity.shape[0]/10000000))
    opacity_gaussians._scaling = mask(-torch.ones_like(gaussians._scaling)*10)
    opacity_gaussians._features_dc = mask(torch.ones_like(gaussians._features_dc))
    opacity_gaussians._features_rest = mask(gaussians._features_rest)

    #mask = lambda l : l[gaussians.get_opacity[:,0] > 0.9]
    #gaussians._xyz = mask(gaussians._xyz)
    #gaussians._rotation = mask(gaussians._rotation)
    #gaussians._scaling = mask(gaussians._scaling)
    #gaussians._features_dc = mask(gaussians._features_dc)
    #gaussians._features_rest = mask(gaussians._features_rest)
    #gaussians._opacity = mask(gaussians._opacity)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start = time.time()
        rendering = torch.clamp(render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"],0.0,1.0)
        end = time.time()
        gt = torch.clamp(view.original_image[0:3, :, :],0.0,1.0)

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        stats["l1_loss"].append(l1_loss(rendering, gt).mean().double().cpu().detach().numpy())
        stats["psnr"].append(psnr(rendering, gt).mean().double().cpu().detach().numpy())
        stats["ssim"].append(ssim(rendering, gt).cpu().detach().numpy())
        stats["lpips"].append(lpips(rendering, gt, net_type='vgg').cpu().detach().numpy())
        stats["fps"].append(1/(end-start))

        density = render(view, density_gaussians, pipeline, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
        opacity_density = render(view, opacity_gaussians, pipeline, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]


        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(density, os.path.join(density_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(opacity_density, os.path.join(opacity_path, '{0:05d}'.format(idx) + ".png"))

    for key in stats.keys():
        stats[key] = np.asarray(stats[key]).mean().item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration),"stats.json"),"w") as f:
        json.dump(stats,f,indent=4)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,overwrite : bool, separate_sh: bool, load_quant: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_quant=load_quant)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp,overwrite, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp,overwrite, separate_sh)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_quant", action="store_true",
                        help='load quantized parameters')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.overwrite, SPARSE_ADAM_AVAILABLE,  args.load_quant)