CUDA_VISIBLE_DEVICES=0 python train.py \
  -s="360_v2/bicycle" \
  -m="output/mini/bicycle" \
  --method "mini" \
  --imp_metric "outdoor" \
  --kmeans_st_iter 10000000000000000 \
  --depth_l1_weight_init 0 \
  --depth_l1_weight_final 0