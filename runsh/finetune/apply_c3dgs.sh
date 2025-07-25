path=$1
load_compact3d_quant=${2:-false}

exp=${path%/*}
ds=${path##*/}

out_path="$exp""/c3dgs/""$ds"

if $load_compact3d_quant; then
    python c3dgs/compress.py -m $path --data_device "cuda" --output_vq $out_path --load_compact3d_quant
else
    python c3dgs/compress.py -m $path --data_device "cuda" --output_vq $out_path
fi
