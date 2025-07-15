dataset="bonsai"

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
  -m="output/mini_opacity/$dataset" \
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
  -m="output/mini_lessdepth_opacity/$dataset" \
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