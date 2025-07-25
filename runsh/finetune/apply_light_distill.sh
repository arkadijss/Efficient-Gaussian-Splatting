
path=$1
dataset_path=$2
teacher_model=$3

exp=${path%/*}
ds=${path##*/}

out_path="$exp""_distill/""$ds"

mkdir -p $out_path

cp -r $path"/point_cloud" $out_path
cp -r $path"/cameras.json" $out_path
cp -r $path"/cfg_args" $out_path
cp -r $path"/exposure.json" $out_path
cp -r $path"/input.ply" $out_path

sed -i "s#$path#$out_path#" $out_path"/cfg_args"

CUDA_VISIBLE_DEVICES=0 python light_gauss/distill_train.py \
    -s "$dataset_path" \
    -m "$out_path" \
    --teacher_model $teacher_model \
    --iteration 10000 \
    --eval \
    --new_max_sh 2 \
    --position_lr_max_steps 10000 \
    --enable_covariance \
    --test_iterations 10000 \
    --save_iterations 10000 \

rm -r $out_path/point_cloud/iteration_5000