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

import sys
import os
sys.path.insert(0, os.path.join(sys.path[0], "../"))


import torch
from light_gauss.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from light_gauss.gaussian_renderer import render
import torchvision
from light_gauss.utils.general_utils import safe_state
from argparse import ArgumentParser,SUPPRESS
from arguments import ModelParams, PipelineParams, get_combined_args
from light_gauss.gaussian_renderer import GaussianModel
import copy
import time
import numpy as np
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss, ssim
import json


def render_set(model_path, name, iteration, views, gaussians, pipeline, background,overwrite):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    density_path = os.path.join(model_path, name, "ours_{}".format(iteration), "density")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    
    if os.path.exists(os.path.join(model_path, name, "ours_{}".format(iteration),"stats.json")) and not overwrite:
        return

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(density_path, exist_ok=True)

    stats = {"l1_loss" : [], "psnr" : [],"ssim" : [],"lpips" : [],"fps" : [], "num_gaussians" : gaussians.get_xyz.shape[0]}

    mask = lambda l : l.cuda()
    density_gaussians = copy.deepcopy(gaussians)
    density_gaussians._xyz = mask(gaussians._xyz)
    density_gaussians._rotation = mask(gaussians._rotation)
    density_gaussians._opacity = mask(-torch.ones_like(gaussians._opacity)*1.0)
    density_gaussians._scaling = mask(-torch.ones_like(gaussians._scaling)*10.0)
    density_gaussians._features_dc = mask(torch.ones_like(gaussians._features_dc)*1.0)
    density_gaussians._features_rest = mask(gaussians._features_rest)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start = time.time()
        rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"],0.0,1.0)
        end = time.time()
        gt = torch.clamp(view.original_image[0:3, :, :],0.0,1.0)

        stats["l1_loss"].append(l1_loss(rendering, gt).mean().double().cpu().detach().numpy())
        stats["psnr"].append(psnr(rendering, gt).mean().double().cpu().detach().numpy())
        stats["ssim"].append(ssim(rendering, gt).cpu().detach().numpy())
        stats["lpips"].append(lpips(rendering, gt, net_type='vgg').cpu().detach().numpy())
        stats["fps"].append(1/(end-start))

        density = render(view, density_gaussians, pipeline, torch.tensor([0.0, 0.0, 0.0],device="cuda"))["render"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(density, os.path.join(density_path, '{0:05d}'.format(idx) + ".png"))

    for key in stats.keys():
        stats[key] = np.asarray(stats[key]).mean().item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration),"stats.json"),"w") as f:
        json.dump(stats,f,indent=4)

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    load_vq: bool, 
    overwrite : bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_vq= load_vq)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                overwrite
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                overwrite
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--load_vq", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.iteration == -10:
        args.iteration = None

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.load_vq,
        args.overwrite
    )
