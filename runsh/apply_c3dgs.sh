path=$1

exp=${path%/*}
ds=${path##*/}

out_path="$exp""/c3dgs/""$ds"

python c3dgs/compress.py -m $path --data_device "cuda" --output_vq $out_path