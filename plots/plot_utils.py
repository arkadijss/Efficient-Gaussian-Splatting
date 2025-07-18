from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from prettytable import PrettyTable
from PIL import Image
import io
import copy
import json
import os

def plot_histograms(super_histograms,scaling=1000,upper_cutoff=None):

    max_num = 0
    min_num = 1000000000000000000000000000000
    steps = []
    for histograms in super_histograms:
        local_steps = []
        for hist in histograms:
            local_steps.append(hist.step)
        if len(local_steps) > len(steps):
            steps = local_steps
    
    for histograms in super_histograms:
        for hist in histograms:
            num = hist.histogram_value.num
            if num > max_num:
                max_num = num
            if num < min_num:
                min_num = num

    fig,axs = plt.subplots(len(steps),len(super_histograms))

    if len(super_histograms) == 1:
        axs = axs[:,None]

    #cmap = plt.cm.viridis 
    #cmap = colors.LinearSegmentedColormap.from_list("viridis_cut",cmap(np.linspace(0.05,0.95,100)))
    cmap = sns.color_palette("blend:green,yellow,orange,red,firebrick",as_cmap=True)
    norm = Normalize(vmin=min_num, vmax=max_num)

    def _plot_histograms(histograms,axs):
        for i,hist in enumerate(histograms):
            step = hist.step
            num = hist.histogram_value.num
            bucket_limit = hist.histogram_value.bucket_limit
            count = hist.histogram_value.bucket

            if upper_cutoff:
                bucket_limit = [bucket_limit[i] for i in range(len(bucket_limit)) if bucket_limit[i]<upper_cutoff]
                count = [count[i] for i in range(len(bucket_limit)) if bucket_limit[i]<upper_cutoff]

            bucket_center = [hist.histogram_value.bucket_limit[0],*[(bucket_limit[i]+bucket_limit[i+1])/2 for i in range(len(bucket_limit)-1)]]
            count_approx = [int(c/scaling) for c in count]
            samples = [e for bc,ca in zip(bucket_center,count_approx) for e in [bc]*ca]
            samples = np.array(samples)

            df = pd.DataFrame({'value': samples,'step': step})
            ax = axs[i]
            
            sns.kdeplot(df,ax=ax,fill=True,warn_singular=False)

            # set color
            ax.collections[0].set_facecolor(cmap(norm(num))) 
            ax.collections[0].set_edgecolor("black") 
            ax.get_legend().remove()
            
        for i,ax in enumerate(axs):
            # make axis transparent
            ax.patch.set_alpha(0) 

            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel("")

            # remove xtick/labels for all but bottom axis
            if i < axs.shape[0]-1:
                ax.set_xticks([])
                ax.set_xticklabels([])

            if i < axs.shape[0]-1:
                ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)


    for i,histograms in enumerate(super_histograms):
        _plot_histograms(histograms,axs[:,i])

    # Make the axes overlap
    #### TAKEN FROM: https://github.com/leotac/joypy ####
    h_pad = 5 + (- 5*(1 + 1))
    fig.tight_layout(h_pad=h_pad)
    #####################################################################################

    # Add the y axis on the far left based on axes positions
    yax = fig.add_axes([axs[-1,0].get_position().x0-0.02, axs[-1,0].get_position().y0, 0.02, axs[0,0].get_position().y0-axs[-1,0].get_position().y0]) 
    yax.yaxis.set_ticks_position('left')
    yax.xaxis.set_visible(False)
    yax.spines['top'].set_visible(False)
    yax.spines['right'].set_visible(False)
    yax.spines['bottom'].set_visible(False)
    yticks = []
    ylabels = []
    for ax,step in zip(axs[:,0],steps):
        yticks.append((ax.get_position().y0-axs[-1,0].get_position().y0)/(axs[0,0].get_position().y0-axs[-1,0].get_position().y0))
        ylabels.append(step)
    yax.set_yticks(yticks)
    yax.set_yticklabels(ylabels)
    
    # Make colormap legend
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    color_legend_ax = fig.add_axes([1,axs[-1,0].get_position().y0+axs[-1,0].get_position().y0, 0.02, axs[0,0].get_position().y0-2*axs[-1,0].get_position().y0])
    fig.colorbar(sm, cax=color_legend_ax)
    
    return fig,axs

def mean_fn(l):
    return sum(l)/len(l)

def info_table(event_accs):
    data_table = PrettyTable(['Exp', 'Num Gaussians', 'Max Mem used (GB)', 'avg itertime (s)','total time (m)','Min total train loss','Min test l1_loss','Max test psnr', 'Max test ssim', 'Min test lpips'])

    for name,event_acc in event_accs.items():
        data_table.add_row([name,
                            event_acc.Scalars("total_points")[-1].value,
                            round(max([event.value for event in event_acc.Scalars("mem")])/1024**3,2),
                            round(sum([event.value for event in event_acc.Scalars("iter_time")])/36000,2),
                            round((event_acc.Scalars("total_points")[-1].wall_time - event_acc.Scalars("total_points")[0].wall_time)/60,2),
                            round(min([event.value for event in event_acc.Scalars('train_loss_patches/total_loss')]),4),
                            round(min([event.value for event in event_acc.Scalars("test/loss_viewpoint - l1_loss")]),4),
                            round(max([event.value for event in event_acc.Scalars("test/loss_viewpoint - psnr")]),4),
                            round(max([event.value for event in event_acc.Scalars("test/loss_viewpoint - ssim")]),4),
                        round(min([event.value for event in event_acc.Scalars("test/loss_viewpoint - lpips")]),4)])
    
    return data_table

def get_zoom_fn(x,y,s):
    return lambda img : img.crop((x-img.width/s,y-img.height/s,x+img.width/s,y+img.height/s)) 

def plot_images(event_accs, idx, img, zoom_fns = [None]):
    fig,axs = plt.subplots(len(zoom_fns),len(list(event_accs.values())))
    if len(zoom_fns) == 1:
        axs = axs[None,:]
    for x,zoom_fn in enumerate(zoom_fns):
        for y,(name,event_acc) in enumerate(event_accs.items()):    
            ax = axs[x,y]
            if zoom_fn:
                ax.imshow(zoom_fn(Image.open(io.BytesIO(event_acc.Images(img)[idx].encoded_image_string))))
            else:
                ax.imshow(Image.open(io.BytesIO(event_acc.Images(img)[idx].encoded_image_string)))

            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
    return fig,axs

def subdict(d,*ks):
    return {k : d[k] for k in ks}

def get_storage(stats,paths):
    pass

def load_stats(eval_paths):
    stats = copy.deepcopy(eval_paths)
    _load_stats(stats)
    return stats

def _load_stats(paths):
    for key,val in paths.items():
        if type(val) == dict:
            _load_stats(val)
        else:
            if os.path.exists(os.path.join(val,"stats.json")):
                with open(os.path.join(val,"stats.json"),"r") as f:
                    paths[key] = json.load(f)
            else:
                paths[key] = {}

def merge_psnr_ssim_lpips(stats):
    for exp in stats.values():
        for compression in exp.values():
            for ds in compression.values():
                if "psnr" in ds.keys() and "ssim" in ds.keys() and "lpips" in ds.keys():
                    ds["psl"] = (ds["psnr"]+ds["ssim"]+(1-ds["lpips"]))/3

def normalize_stats(stats,datasets):
    min_max = {ds : {} for ds in datasets}
    _get_normalization_stats(stats,min_max)
    refined_stats = copy.deepcopy(stats)
    _set_normalized_stats(refined_stats,min_max)
    return refined_stats,min_max

    
def _get_normalization_stats(stats,min_max):
    for key,val in stats.items():
        if key in min_max.keys():
            if type(val) == dict:
                for stat_key,stat_val in val.items():
                    if not stat_key in min_max[key].keys():
                        min_max[key][stat_key] = {"min" : stat_val, "max" : stat_val}
                    else:
                        if stat_val < min_max[key][stat_key]["min"]:
                            min_max[key][stat_key]["min"] = stat_val
                        if stat_val > min_max[key][stat_key]["max"]:
                            min_max[key][stat_key]["max"] = stat_val
        else:
            _get_normalization_stats(val,min_max)

def _set_normalized_stats(stats,min_max):
    for key,val in stats.items():
        if key in min_max.keys():
            if type(val) == dict:
                for stat_key,stat_val in val.items():
                    val[stat_key] = (stat_val-min_max[key][stat_key]["min"])/(min_max[key][stat_key]["max"]-min_max[key][stat_key]["min"])
        else:
            _set_normalized_stats(val,min_max)


def x_vs_y(xs1,ys1,xs2,ys2,l1,l2,fig,axs):
    sns.scatterplot(x=[*xs1,*xs2],y=[*ys1,*ys2],hue=[*[l1 for _ in range(len(xs1))],*[l2 for _ in range(len(xs2))]],ax=axs)
    sns.lineplot(x=[*xs1,*xs2],y=[*ys1,*ys2],hue=[*[l1 for _ in range(len(xs1))],*[l2 for _ in range(len(xs2))]],ax=axs,legend=False)

    axs.grid(True)