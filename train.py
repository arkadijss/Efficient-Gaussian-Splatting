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
import pdb
from os.path import join
import datetime
import json
import time
from bitarray import bitarray

import numpy as np
import torch
from random import randint
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
from methods import method_handles, Method
from scene.kmeans_quantize import Quantize_kMeans
import faiss


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    method: Method = method_handles[pipe.method](gaussians, opt, args)

    method.init_mask_blur()

    #################################################### Comp 3dgs ####################################################
    num_gaussians_per_iter = []
    # k-Means quantization
    quantized_params = args.quant_params
    n_cls = args.kmeans_ncls
    n_cls_sh = args.kmeans_ncls_sh
    n_cls_dc = args.kmeans_ncls_dc
    n_it = args.kmeans_iters
    kmeans_st_iter = args.kmeans_st_iter
    freq_cls_assn = args.kmeans_freq
    if 'pos' in quantized_params:
        kmeans_pos_q = Quantize_kMeans(num_clusters=n_cls_dc, num_iters=n_it)
    if 'dc' in quantized_params:
        kmeans_dc_q = Quantize_kMeans(num_clusters=n_cls_dc, num_iters=n_it)
    if 'sh' in quantized_params:
        kmeans_sh_q = Quantize_kMeans(num_clusters=n_cls_sh, num_iters=n_it)
    if 'scale' in quantized_params:
        kmeans_sc_q = Quantize_kMeans(num_clusters=n_cls, num_iters=n_it)
    if 'rot' in quantized_params:
        kmeans_rot_q = Quantize_kMeans(num_clusters=n_cls, num_iters=n_it)
    if 'scale_rot' in quantized_params:
        kmeans_scrot_q = Quantize_kMeans(num_clusters=n_cls, num_iters=n_it)
    if 'sh_dc' in quantized_params:
        kmeans_shdc_q = Quantize_kMeans(num_clusters=n_cls_sh, num_iters=n_it)
    ###################################################################################################################

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = method.render(custom_cam, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        method.update_learning_rate_and_sh_degree(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        #################################################### Comp 3dgs ####################################################
        # quantize params
        if iteration > kmeans_st_iter:
            if iteration % freq_cls_assn == 1:
                assign = True
            else:
                assign = False
            if 'pos' in quantized_params:
                kmeans_pos_q.forward_pos(gaussians, assign=assign)
            if 'dc' in quantized_params:
                kmeans_dc_q.forward_dc(gaussians, assign=assign)
            if 'sh' in quantized_params:
                kmeans_sh_q.forward_frest(gaussians, assign=assign)
            if 'scale' in quantized_params:
                kmeans_sc_q.forward_scale(gaussians, assign=assign)
            if 'rot' in quantized_params:
                kmeans_rot_q.forward_rot(gaussians, assign=assign)
            if 'scale_rot' in quantized_params:
                kmeans_scrot_q.forward_scale_rot(gaussians, assign=assign)
            if 'sh_dc' in quantized_params:
                kmeans_shdc_q.forward_dcfrest(gaussians, assign=assign)
        ###################################################################################################################

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = method.render(viewpoint_cam, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        #################################################### Comp 3dgs ####################################################
        if args.opacity_reg:
            if iteration > args.max_prune_iter or iteration < 15000:
                lambda_reg = 0.
            else:
                lambda_reg = args.lambda_reg
            L_reg_op = gaussians.get_opacity.sum()
            loss += lambda_reg * L_reg_op
        ###################################################################################################################

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, method.render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                #################################################### Comp 3dgs ####################################################
                # Save only the non-quantized parameters in ply file.
                all_attributes = {'xyz': 'xyz', 'dc': 'f_dc', 'sh': 'f_rest', 'opacities': 'opacities',
                                  'scale': 'scale', 'rot': 'rotation'}
                save_attributes = [val for (key, val) in all_attributes.items() if key not in quantized_params]
                if iteration > kmeans_st_iter:
                    scene.save(iteration, save_q=quantized_params, save_attributes=save_attributes)
                    
                    # Save indices and codebook for quantized parameters
                    kmeans_dict = {'rot': kmeans_rot_q, 'scale': kmeans_sc_q, 'sh': kmeans_sh_q, 'dc': kmeans_dc_q}
                    kmeans_list = []
                    for param in quantized_params:
                        kmeans_list.append(kmeans_dict[param])
                    out_dir = join(scene.model_path, 'point_cloud/iteration_%d' % iteration)
                    save_kmeans(kmeans_list, quantized_params, out_dir)
                else:
                    scene.save(iteration, save_q=[])
                ###################################################################################################################

            # Densification
            if iteration < opt.densify_until_iter:
                method.densify(visibility_filter, scene, iteration, viewspace_point_tensor, radii, dataset, render_pkg, image, pipe, background)

            #################################################### Comp 3dgs ####################################################
            # Prune Gaussians every 1000 iterations from iter 15000 to max_prune_iter if using opacity regularization
            if args.opacity_reg and iteration > 15000:
                if iteration <= args.max_prune_iter and iteration % 1000 == 0:
                    print('Num Gaussians: ', gaussians._xyz.shape[0])
                    size_threshold = None
                    gaussians.prune(0.005, scene.cameras_extent, size_threshold, radii)
                    print('Num Gaussians after prune: ', gaussians._xyz.shape[0])
            ###################################################################################################################

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

#################################################### Comp 3dgs ####################################################
def dec2binary(x, n_bits=None):
    """Convert decimal integer x to binary.

    Code from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    """
    if n_bits is None:
        n_bits = torch.ceil(torch.log2(x)).type(torch.int64)
    mask = 2**torch.arange(n_bits-1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)

def save_kmeans(kmeans_list, quantized_params, out_dir):
    """Save the codebook and indices of KMeans.

    """
    # Convert to bitarray object to save compressed version
    # saving as npy or pth will use 8bits per digit (or boolean) for the indices
    # Convert to binary, concat the indices for all params and save.
    bitarray_all = bitarray([])
    for kmeans in kmeans_list:
        n_bits = int(np.ceil(np.log2(len(kmeans.cls_ids))))
        assignments = dec2binary(kmeans.cls_ids, n_bits)
        bitarr = bitarray(list(assignments.cpu().numpy().flatten()))
        bitarr = bitarray(list(assignments.cpu().numpy().flatten()))
        bitarray_all.extend(bitarr)
    with open(join(out_dir, 'kmeans_inds.bin'), 'wb') as file:
        bitarray_all.tofile(file)

    # Save details needed for loading
    args_dict = {}
    args_dict['params'] = quantized_params
    args_dict['n_bits'] = n_bits
    args_dict['total_len'] = len(bitarray_all)
    np.save(join(out_dir, 'kmeans_args.npy'), args_dict)
    centers_dict = {param: kmeans.centers for (kmeans, param) in zip(kmeans_list, quantized_params)}

    # Save codebook
    torch.save(centers_dict, join(out_dir, 'kmeans_centers.pth'))
###################################################################################################################

def prepare_output_and_logger(data_args,opt_args,pipe_args):    
    if not data_args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        data_args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(data_args.model_path))
    os.makedirs(data_args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(data_args))))

    with open(os.path.join(data_args.model_path, "args.json"), 'w') as f:
        json.dump({"args": vars(args),"dataset_args": vars(data_args),"opt_args": vars(opt_args),"pipe_args": vars(pipe_args)},f,indent=4)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(data_args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('mem', torch.cuda.memory_allocated(0), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims = []
                lpipss = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image, net_type='vgg'))                    

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 

                ssims_test=torch.tensor(ssims).mean()
                lpipss_test=torch.tensor(lpipss).mean()

                print("\n[ITER {}] Evaluating {}: ".format(iteration, config['name']))
                print("  L1   : {:>12.7f}".format(l1_test, ".5"))
                print("  PSNR : {:>12.7f}".format(psnr_test.mean(), ".5"))
                print("  SSIM : {:>12.7f}".format(ssims_test.mean(), ".5"))
                print("  LPIPS : {:>12.7f}".format(lpipss_test.mean(), ".5"))
                print("")
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssims_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpipss_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/size_histogram", torch.linalg.norm(scene.gaussians.get_scaling,ord=2,dim=-1), iteration)
            x = scene.gaussians.get_xyz.cpu().numpy().astype('float32')
            perm = np.random.permutation(np.arange(x.shape[0]))
            inv_perm = np.argsort(perm)

            x = x[perm,:]
            x = np.array_split(x,max(int(x.shape[0]/400000),1))

            distances = []
            for e in x:
                cpu_index = faiss.IndexFlatL2(3) 
                gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)  

                gpu_index.add(e) 

                dist, _ = gpu_index.search(e, 1)
                distances.append(dist[:,0])

            distances = np.concatenate(distances)
            # not neccessary but feels wrong not to do
            distances = distances[inv_perm]
            tb_writer.add_histogram("scene/cdist_min_histogram", torch.min(cdist,dim=-1), iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[int(1000*i) for i in range(31)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--simp_iteration1", type=int, default = 15_000)
    parser.add_argument("--simp_iteration2", type=int, default = 20_000)
    parser.add_argument("--num_depth", type=int, default = 3_500_000)
    parser.add_argument("--num_max", type=int, default = 4_500_000)
    parser.add_argument("--sampling_factor", type=float, default = 0.5)

    parser.add_argument("--imp_metric", required=True, type=str, default = None)
    
    #################################################### Comp 3dgs ####################################################
    # Compress3D parameters
    parser.add_argument('--kmeans_st_iter', type=int, default=60000,
                        help='Start k-Means based vector quantization from this iteration')
    parser.add_argument('--kmeans_ncls', type=int, default=4096,
                        help='Number of clusters in k-Means quantization')
    parser.add_argument('--kmeans_ncls_sh', type=int, default=4096,
                        help='Number of clusters in k-Means quantization of spherical harmonics')
    parser.add_argument('--kmeans_ncls_dc', type=int, default=4096,
                        help='Number of clusters in k-Means quantization of DC component of color')
    parser.add_argument('--kmeans_iters', type=int, default=1,
                        help='Number of assignment and centroid calculation iterations in k-Means')
    parser.add_argument('--kmeans_freq', type=int, default=100,
                        help='Frequency of cluster assignment in k-Means')
    parser.add_argument('--grad_thresh', type=float, default=0.0002,
                        help='threshold on xyz gradients for densification')
    parser.add_argument("--quant_params", nargs="+", type=str, default=['sh', 'dc', 'scale', 'rot'])

    # Opacity regularization parameters
    parser.add_argument('--max_prune_iter', type=int, default=20000,
                        help='Iteration till which pruning is done')
    parser.add_argument('--opacity_reg', action='store_true', default=False,
                        help='use opacity regularization during training')
    parser.add_argument('--lambda_reg', type=float, default=0.,
                        help='Weight for opacity regularization in loss')
    ###################################################################################################################

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
