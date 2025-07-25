path=$1
dataset_path=$2
load_compact3d_quant=${3:-false}

exp=${path%/*}
ds=${path##*/}

out_path="$exp""/mini_c/""$ds"

mkdir -p $out_path

cp -r $path"/point_cloud" $out_path
cp -r $path"/cameras.json" $out_path
cp -r $path"/cfg_args" $out_path
cp -r $path"/exposure.json" $out_path
cp -r $path"/input.ply" $out_path

sed -i "s#$path#$out_path#" $out_path"/cfg_args"

if $load_compact3d_quant; then
    python mini_splatting_compress/run.py -m $out_path -s $dataset_path --load_comapct3d_quant
else
    python mini_splatting_compress/run.py -m $out_path -s $dataset_path
fi