path_base=./

ncls=32768
ncls_sh=4096
ncls_dc=4096
kmeans_iters=1
st_iter=20000
max_iters=30000
max_prune_iter=20000
lambda_reg=1e-7
cuda_device=0

CUDA_VISIBLE_DEVICES=$cuda_device python train.py \
  -s="360_v2/bicycle" \
  -m="output/compact3d/bicycle" \
  --imp_metric "outdoor" \
  --kmeans_ncls "$ncls" \
  --kmeans_ncls_sh "$ncls_sh" \
  --kmeans_ncls_dc "$ncls_dc" \
  --kmeans_st_iter "$st_iter" \
  --kmeans_iters "$kmeans_iters" \
  --quant_params sh dc rot scale\
  --opacity_reg \
  --kmeans_freq 100 \
  --lambda_reg "$lambda_reg" \
  --max_prune_iter "$max_prune_iter" \
  --eval \
  --depth_l1_weight_init 0 \
  --depth_l1_weight_final 0
