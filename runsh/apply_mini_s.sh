path=$1
dataset_path=$2

exp=${path%/*}
ds=${path##*/}

out_path="$exp""/mini_s/""$ds"

mkdir -p $out_path

cp -r $path"/point_cloud" $out_path
cp -r $path"/cameras.json" $out_path
cp -r $path"/cfg_args" $out_path
cp -r $path"/exposure.json" $out_path
cp -r $path"/input.ply" $out_path

sed -i "s#$path#$out_path#" $out_path"/cfg_args"

python mini_splatting_compress/run.py -m $out_path -s $dataset_path

python mini_splatting_compress/render.py -m $out_path -s $dataset_path --overwrite