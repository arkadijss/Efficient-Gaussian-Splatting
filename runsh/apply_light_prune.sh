
path=$1
dataset_path=$2
load_compact3d_quant=$3

prune_percent=(${4:-0.66})
prune_decay=(${5:-1})
v_pow=(${6:-0.1})

declare -a prune_type=${7:-"v_important_score"}

exp=${path%/*}
ds=${path##*/}

out_path="$exp""/light_prune_"$prune_percent"_"$prune_decay"_"$v_pow"_"$prune_type"/""$ds"

mkdir -p $out_path

cp -r $path"/cameras.json" $out_path
cp -r $path"/cfg_args" $out_path
cp -r $path"/exposure.json" $out_path
cp -r $path"/input.ply" $out_path

sed -i "s#$path#$out_path#" $out_path"/cfg_args"

if $load_compact3d_quant; then
    point_cloud_path=$exp/$ds/point_cloud/iteration_30000/point_cloud_decompressed.ply
else
    point_cloud_path=$exp/$ds/point_cloud/iteration_30000/point_cloud.ply
fi

CUDA_VISIBLE_DEVICES=0 python light_gauss/prune_finetune.py \
    -s $dataset_path \
    -m $out_path \
    --start_pointcloud $point_cloud_path \
    --eval \
    --iteration 5000 \
    --prune_percent $prune_percent \
    --prune_type $prune_type \
    --prune_decay $prune_decay \
    --position_lr_max_steps 5000 \
    --test_iterations 5000 \
    --save_iterations 5000 \
    --prune_iterations 2 \
    --v_pow $v_pow