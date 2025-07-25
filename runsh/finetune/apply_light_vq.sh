
path=$1

exp=${path%/*}
ds=${path##*/}

out_path="$exp""_vq/""$ds"


mkdir -p $out_path

cp -r $path"/cameras.json" $out_path
cp -r $path"/cfg_args" $out_path
cp -r $path"/exposure.json" $out_path
cp -r $path"/input.ply" $out_path

sed -i "s#$path#$out_path#" $out_path"/cfg_args"

VQ_RATIO=0.6
CODEBOOK_SIZE=8192

python light_gauss/vectree/vectree.py \
    --input_path "$path/point_cloud/iteration_10000/point_cloud.ply" \
    --save_path ${out_path} \
    --vq_ratio ${VQ_RATIO} \
    --codebook_size ${CODEBOOK_SIZE} \
    --no_IS


#mkdir -p $out_path/point_cloud
#mkdir -p $out_path/point_cloud/iteration_10000
#mv $out_path/extreme_saving $out_path/point_cloud/iteration_10000
#mv $out_path/extreme_saving.zip $out_path/point_cloud/iteration_10000