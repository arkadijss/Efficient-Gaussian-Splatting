from gaussian_renderer import render, render_imp, render_depth
from scene import Scene, GaussianModel
import torch
import numpy as np
from utils.sh_utils import SH2RGB
from abc import ABC, abstractmethod
from arguments import OptimizationParams


def init_cdf_mask(importance, thres=1.0):
    importance = importance.flatten()   
    if thres!=1.0:
        percent_sum = thres
        vals,idx = torch.sort(importance+(1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance>split_val_nonprune 
    else: 
        non_prune_mask = torch.ones_like(importance).bool()
        
    return non_prune_mask


class Method(ABC):
    def __init__(self, gaussians: GaussianModel, opt: OptimizationParams, args):
        self.gaussians = gaussians
        self.opt = opt
        self.args = args

    def init_mask_blur(self):
        pass
    
    @abstractmethod
    def update_learning_rate(self, iteration: int):
        pass

    @abstractmethod
    def oneupSHdegree(self):
        pass

    def update_learning_rate_and_sh_degree(self, iteration: int):
        self.update_learning_rate(iteration)
        self.oneupSHdegree(iteration)

    @abstractmethod
    def render(self, viewpoint_camera, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
        pass
    
    @abstractmethod
    def densify(self, visibility_filter: torch.Tensor, scene: Scene, iteration: int, viewspace_point_tensor: torch.Tensor, radii: torch.Tensor, dataset, render_pkg, image, pipe, background):
        pass


class GaussianSplatting(Method):
    def update_learning_rate(self, iteration: int):
        self.gaussians.update_learning_rate(iteration)

    def oneupSHdegree(self, iteration: int):
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

    def render(self, viewpoint_camera, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
        return render(viewpoint_camera, self.gaussians, pipe, bg_color, scaling_modifier, separate_sh, override_color, use_trained_exp)
    
    def densify(self, visibility_filter: torch.Tensor, scene: Scene, iteration: int, viewspace_point_tensor: torch.Tensor, radii: torch.Tensor, dataset, render_pkg, image, pipe, background):
        # Keep track of max radii in image-space for pruning
        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
            size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
            self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
        
        if iteration % self.opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == self.opt.densify_from_iter):
            self.gaussians.reset_opacity()

        return False

class MiniSplatting(Method):
    def init_mask_blur(self):
        self.mask_blur = torch.zeros(self.gaussians._xyz.shape[0], device='cuda')
    
    def update_learning_rate(self, iteration: int):
        if iteration<self.args.simp_iteration1:
            self.gaussians.update_learning_rate(iteration)
        else:
            self.gaussians.update_learning_rate(iteration-self.args.simp_iteration1+5000)

    def oneupSHdegree(self, iteration: int):
        if iteration % 1000 == 0 and iteration>self.args.simp_iteration1:
            self.gaussians.oneupSHdegree()
    
    def render(self, viewpoint_camera, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
        return render_imp(viewpoint_camera, self.gaussians, pipe, bg_color, scaling_modifier)
    
    def densify(self, visibility_filter: torch.Tensor, scene: Scene, iteration: int, viewspace_point_tensor: torch.Tensor, radii: torch.Tensor, dataset, render_pkg, image, pipe, background):
        reset_viewpoint_stack = False
        # Keep track of max radii in image-space for pruning
        gaussians = self.gaussians
        args = self.args
        opt = self.opt
        mask_blur = self.mask_blur
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        area_max = render_pkg["area_max"]
        #print(f"Gauss: {self.gaussians._xyz.shape} | mask blur {mask_blur.shape} | area max {area_max.shape}")

        mask_blur = torch.logical_or(mask_blur, area_max>(image.shape[1]*image.shape[2]/5000))

        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration % 5000!=0 and gaussians._xyz.shape[0]<args.num_max:  
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussians.densify_and_prune_split(opt.densify_grad_threshold, 
                                            0.005, scene.cameras_extent, 
                                            size_threshold, mask_blur, radii)
            self.mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
            
        if iteration%5000==0:
            out_pts_list=[]
            gt_list=[]
            views=scene.getTrainCameras()
            for view in views:
                gt = view.original_image[0:3, :, :]
                render_depth_pkg = render_depth(view, gaussians, pipe, background)
                out_pts = render_depth_pkg["out_pts"]
                accum_alpha = render_depth_pkg["accum_alpha"]

                prob=1-accum_alpha

                prob = prob/prob.sum()
                prob = prob.reshape(-1).cpu().numpy()

                factor=1/(image.shape[1]*image.shape[2]*len(views)/args.num_depth)

                N_xyz=prob.shape[0]
                num_sampled=int(N_xyz*factor)

                indices = np.random.choice(N_xyz, size=num_sampled, 
                                        p=prob,replace=False)
                
                out_pts = out_pts.permute(1,2,0).reshape(-1,3)
                gt = gt.permute(1,2,0).reshape(-1,3)

                out_pts_list.append(out_pts[indices])
                gt_list.append(gt[indices])       

            out_pts_merged=torch.cat(out_pts_list)
            gt_merged=torch.cat(gt_list)

            gaussians.reinitial_pts(out_pts_merged, gt_merged)
            gaussians.training_setup(opt)
            self.mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
            torch.cuda.empty_cache()
            reset_viewpoint_stack = True

        if iteration == args.simp_iteration1:
            imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
            accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
            views=scene.getTrainCameras()
            for view in views:
                # print(idx)
                render_pkg = render_imp(view, gaussians, pipe, background)
                accum_weights = render_pkg["accum_weights"]
                area_proj = render_pkg["area_proj"]
                area_max = render_pkg["area_max"]

                accum_area_max = accum_area_max+area_max

                if args.imp_metric=='outdoor':
                    mask_t=area_max!=0
                    temp=imp_score+accum_weights/area_proj
                    imp_score[mask_t] = temp[mask_t]
                else:
                    imp_score=imp_score+accum_weights
            
            imp_score[accum_area_max==0]=0
            prob = imp_score/imp_score.sum()
            prob = prob.cpu().numpy()

            factor=args.sampling_factor
            N_xyz=gaussians._xyz.shape[0]
            num_sampled=int(N_xyz*factor*((prob!=0).sum()/prob.shape[0]))
            indices = np.random.choice(N_xyz, size=num_sampled, 
                                    p=prob, replace=False)

            mask = np.zeros(N_xyz, dtype=bool)
            mask[indices] = True

            # print(mask.sum(), mask.sum()/mask.shape[0])                 
            gaussians.prune_points(mask==False) 
            
            gaussians.max_sh_degree=dataset.sh_degree
            gaussians.reinitial_pts(gaussians._xyz, 
                                SH2RGB(gaussians._features_dc+0)[:,0])
            
            gaussians.training_setup(opt)
            torch.cuda.empty_cache()
            reset_viewpoint_stack = True

        if iteration == args.simp_iteration2:
            imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
            accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
            views=scene.getTrainCameras()
            for view in views:
                gt = view.original_image[0:3, :, :]
                
                render_pkg = render_imp(view, gaussians, pipe, background)
                accum_weights = render_pkg["accum_weights"]
                area_proj = render_pkg["area_proj"]
                area_max = render_pkg["area_max"]

                accum_area_max = accum_area_max+area_max

                if args.imp_metric=='outdoor':
                    mask_t=area_max!=0
                    temp=imp_score+accum_weights/area_proj
                    imp_score[mask_t] = temp[mask_t]
                else:
                    imp_score=imp_score+accum_weights
                
            imp_score[accum_area_max==0]=0
            non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99) 
                            
            gaussians.prune_points(non_prune_mask==False)
            gaussians.training_setup(opt)
            torch.cuda.empty_cache()   

        return reset_viewpoint_stack


method_handles = {
    "orig": GaussianSplatting,
    "mini": MiniSplatting,
}
