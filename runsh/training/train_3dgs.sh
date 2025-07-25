dataset_path=$1
dataset=$2

CUDA_VISIBLE_DEVICES=0 python train.py \
  -s="$dataset_path/$dataset" \
  -m="output/orig/$dataset" \
  --imp_metric "outdoor" \
  --kmeans_st_iter 10000000000000000 \
  --depth_l1_weight_init 0 \
  --depth_l1_weight_final 0 \
  --eval