# 3D Gaussian Splatting and Compression for Efficient and Accelerated Novel View Synthesis

This repository contains the implementation of the project "3D Gaussian Splatting and Compression for
Efficient and Accelerated Novel View Synthesis" for the course "Hands-on AI based 3D Vision" (summer semester 2025) at the University of Tübingen. The project authors are Arkādijs Sergejevs and Leon Lemke.

## Layout

The datasets are in ./datasets. Plots can be found in ./plots. Our data can be requested via leon.lemke@student.uni-tuebingen.de. The data is in the following format:

```
output
│   
└─── Training Setting
│   │
│   └─── Scene Name
│   │   │
│   │   └─── files.file
│   │
│   └─── Finetune Setting
│       │
│       └─── Scene Name
│           │
│           └─── files.file
```

## Environment

3d Gaussian Splatting, Mini-Splatting, Compact Gaussian Splatting and Light Gaussian can all be run from the same environment:
1. Install environment.yml
2. Additionally install the dependencies from ./light_gauss/environment.yml
3. Install submodules/diff-gaussian-rasterization_ms
4. Install bitarray and tensorboard

Compressed Gaussian Splatting requires a different CUDA Toolkit (12.1) version:
1. Install ./c3dgs/environment.yml

For the specifics, please visit the respective repo. Their installation guide, especially on which CUDA versions are neccessary and on bugs that might occur is much more in depth. Otherwise ChatGPT can be a great installation helper. 

## Training
Original 3d gaussian Splatting, Mini Splatting and Compact Gaussian Splatting can be run via ./train.py. Example bash scripts for that can be found in ./runsh/training.


## Finetuning
The code for Compressed Gaussian Splatting can be found in ./c3dgs. The code for Light Gaussian Splatting can be found in ./light_gauss. The code for Mini Splatting Compression can be found in ./mini_splatting_compress. Example bash scripts for that can be found in ./runsh/finetune.
Note that when using a Compact Gaussian Splatting checkpoint it has to be decompressed via ./compatc3d_decompress/decompress_to_ply.py first.

## Rendering

The following table shows which render.py should be used for which checkpoint:

Training/Finetune | Checkpoint | render.py | Notes |
| -------- | -------- | ------- | ------- |
| Training | 3d Gaussian Splatting | ./render.py |  |
| Training | Mini Splatting | ./render.py |  |
| Training | Compact Gaussian Splatting | ./render.py | Specify --load_quant |
| Finetune | Compressed Gaussian Splatting | ./c3dgs/render.py |  |
| Finetune | Mini Splatting Compression | ./mini_splatting_compress/render.py |  |
| Finetune | Light Gaussian Pruning | ./light_gauss/render.py |  |
| Finetune | Light Gaussian Pruning + Distill + VQ | ./light_gauss/render.py | Specify --load_vq |


## Acknowledgments

This project and most of the code in this repo is based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Mini-Splatting](https://github.com/fatPeter/mini-splatting),[Compact Gaussian Splatting](https://github.com/UCDvision/compact3d/tree/main), [Compressed Gaussian Splatting](https://github.com/KeKsBoTer/c3dgs) and [Light Gaussian Splatting](https://github.com/VITA-Group/LightGaussian). Our contribution is mainly merging Mini Splatting and Compact Gaussian Splatting into the newest 3dgs repo version. However, in some places we had to modify small sections of code or perform bug fixes. We additionally thank ChatGPT for helping with weird CUDA installation issues, as well as [Joypy](https://github.com/leotac/joypy), [torch](https://pytorch.org/), [numpy](https://numpy.org/) and [matplotlib](https://matplotlib.org/).
