dataset_path=$1
dataset=$2

CUDA_VISIBLE_DEVICES=0 python train.py \
  -s="$dataset_path/$dataset" \
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