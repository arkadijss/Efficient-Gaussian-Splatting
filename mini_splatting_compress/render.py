import sys
import os
sys.path.insert(0, os.path.join(sys.path[0], "../"))


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
import numpy as np

from Haar3D_torch import haar3D, inv_haar3D
from utils.sh_utils import SH2RGB
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
import json
from render import render_set


try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False




def decompress(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, overwrite : bool, separate_sh : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        dir_path=os.path.join(scene.model_path,
                    "point_cloud",
                    "iteration_" + str(iteration),
                    "compressed")

        data = np.load(dir_path+'/compressed_gs.npz')

        pos=data['arr_0']
        CT_q=data['arr_1']
        Qstep=data['arr_2']
        depth=data['arr_3']

        pos=torch.tensor(pos).cuda()
        CT_q=torch.tensor(CT_q).cuda()
        Qstep=torch.tensor(Qstep).cuda()


        # voxlize
        pos_voxlized=(pos-pos.min())/(pos.max()-pos.min())
        pos_voxlized=torch.round(pos_voxlized*(2**depth-1))
        pos_voxlized, pos_idx = np.unique(pos_voxlized.detach().cpu().numpy(), axis=0, return_index=True)     

        # inverse RAHT
        feat_rec = inv_haar3D(pos_voxlized, CT_q*(Qstep).item(), int(depth))

        num_g_voxlized=pos_voxlized.shape[0]


        gaussians._xyz = pos
        gaussians._features_dc = feat_rec[:, :3].reshape(num_g_voxlized, -1, 3).float()
        gaussians._features_rest = feat_rec[:, 3:48].reshape(num_g_voxlized, -1, 3)

        gaussians._scaling = feat_rec[:, 48:51].reshape(num_g_voxlized, 3).float()
        gaussians._rotation = feat_rec[:, 51:55].reshape(num_g_voxlized, 4).float()
        gaussians._opacity = feat_rec[:, 55:56].reshape(num_g_voxlized, 1).float()
        gaussians.active_sh_degree = dataset.sh_degree

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp,overwrite, separate_sh)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30_000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)

    safe_state(False)

    decompress(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.overwrite, SPARSE_ADAM_AVAILABLE)