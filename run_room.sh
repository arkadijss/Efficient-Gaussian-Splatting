dataset="room"

CUDA_VISIBLE_DEVICES=0 python train.py \
  -s="360_v2/$dataset" \
  -m="output/orig/$dataset" \
  --imp_metric "outdoor" \
  --kmeans_st_iter 10000000000000000 \
  --depth_l1_weight_init 0 \
  --depth_l1_weight_final 0 \
  --eval

CUDA_VISIBLE_DEVICES=0 python train.py \
  -s="360_v2/$dataset" \
  -m="output/orig_depth_reg/$dataset" \
  --imp_metric "outdoor" \
  --kmeans_st_iter 10000000000000000 \
  --eval

CUDA_VISIBLE_DEVICES=0 python train.py \
  -s="360_v2/$dataset" \
  -m="output/minsci/$dataset" \
  --method "mini" \
  --imp_metric "outdoor" \
  --kmeans_st_iter 10000000000000000 \
  --depth_l1_weight_init 0 \
  --depth_l1_weight_final 0 \
  --eval


CUDA_VISIBLE_DEVICES=0 python train.py \
  -s="360_v2/$dataset" \
  -m="output/compact3d/$dataset" \
  --imp_metric "outdoor" \
  --kmeans_ncls "32768" \
  --kmeans_ncls_sh "4096" \
  --kmeans_ncls_dc "4096" \
  --kmeans_st_iter "20000" \
  --kmeans_iters "1" \
  --quant_params sh dc rot scale\
  --opacity_reg \
  --kmeans_freq 100 \
  --lambda_reg "1e-7" \
  --max_prune_iter "20000" \
  --depth_l1_weight_init 0 \
  --depth_l1_weight_final 0 \
  --eval \

CUDA_VISIBLE_DEVICES=0 python train.py \
  -s="360_v2/$dataset" \
  -m="output/mini_compact3d/$dataset" \
  --method "mini" \
  --imp_metric "outdoor" \
  --kmeans_ncls "32768" \
  --kmeans_ncls_sh "4096" \
  --kmeans_ncls_dc "4096" \
  --kmeans_st_iter "20000" \
  --kmeans_iters "1" \
  --quant_params sh dc rot scale\
  --opacity_reg \
  --kmeans_freq 100 \
  --lambda_reg "1e-7" \
  --max_prune_iter "20000" \
  --depth_l1_weight_init 0 \
  --depth_l1_weight_final 0 \
  --eval \

CUDA_VISIBLE_DEVICES=0 python train.py \
  -s="360_v2/$dataset" \
  -m="output/mini_lessdepth/$dataset" \
  --method "mini" \
  --imp_metric "outdoor" \
  --kmeans_st_iter 10000000000000000 \
  --num_depth 1000000 \
  --depth_l1_weight_init 0 \
  --depth_l1_weight_final 0 \
  --eval


CUDA_VISIBLE_DEVICES=0 python train.py \
  -s="360_v2/$dataset" \
  -m="output/mini_lessdepth_opacity/$dataset" \
  --method "mini" \
  --imp_metric "outdoor" \
  --kmeans_st_iter 10000000000000000 \
  --num_depth 1000000 \
  --opacity_reg \
  --lambda_reg "1e-7" \
  --max_prune_iter "20000" \
  --depth_l1_weight_init 0 \
  --depth_l1_weight_final 0 \
  --eval


CUDA_VISIBLE_DEVICES=0 python train.py \
  -s="360_v2/$dataset" \
  -m="output/mini_opacity/$dataset" \
  --method "mini" \
  --imp_metric "outdoor" \
  --kmeans_st_iter 10000000000000000 \
  --opacity_reg \
  --lambda_reg "1e-7" \
  --max_prune_iter "20000" \
  --depth_l1_weight_init 0 \
  --depth_l1_weight_final 0 \
  --eval

CUDA_VISIBLE_DEVICES=0 python train.py \
  -s="360_v2/$dataset" \
  -m="output/orig_opacity/$dataset" \
  --method "orig" \
  --imp_metric "outdoor" \
  --kmeans_st_iter 10000000000000000 \
  --opacity_reg \
  --lambda_reg "1e-7" \
  --max_prune_iter "20000" \
  --depth_l1_weight_init 0 \
  --depth_l1_weight_final 0 \
  --eval


