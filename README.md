# 3D Gaussian Splatting and Compression for Efficient and Accelerated Novel View Synthesis

This repository contains the implementation of the project "3D Gaussian Splatting and Compression for
Efficient and Accelerated Novel View Synthesis" for the course "Hands-on AI based 3D Vision" (summer semester 2025) at the University of Tübingen.

# Layout

The datasets are in ./datasets. Plots can be found in ./plots. Our final data can be downloaded from: "".

## Training
Original 3d gaussian Splatting, Mini Splatting and Compact Gaussian Splatting can be run via ./train.py. Example bash scripts for that can be found in ./runsh/training


## Finetuning
The code for Compressed Gaussian Splatting can be found in ./c3dgs. The code for Light Gaussian Splatting can be found in ./light_gauss. The code for Mini Splatting Compression can be found in ./mini_splatting_compress. Example bash scripts for that can be found in ./runsh/finetune.
Note that when finetuning or compression a Compact Gaussian Splatting checkpoint it has to be decompressed via ./compatc3d_decompress/decompress_to_ply.py

## Rendering

The following table shows which render.py should be used for which checkpoint:

Training/Finetune | Checkpoint | render.py | Notes |
| -------- | -------- | ------- | ------- |
| Training | 3dgs | ./render.py |  |
| Training | Mini Splatting | ./render.py |  |
| Training | Compact Gaussian Splatting | ./render.py | Specify --load_quant |
| Finetune | Compressed Gaussian Splatting | ./c3dgs/render.py |  |
| Finetune | Mini Splatting Compression | ./mini_splatting_compress/render.py |  |
| Finetune | Light Gaussian Pruning | ./light_gauss/render.py |  |
| Finetune | Light Gaussian Pruning + Distill + VQ | ./light_gauss/render.py | Specify --load_quant |




The project authors are Arkādijs Sergejevs and Leon Lemke.

## Acknowledgments

This project is based on the great works [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Mini-Splatting](https://github.com/fatPeter/mini-splatting).
