#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "usage: $0 <binary> <input> <output> <width> <height> <model>" >&2
    exit 2
fi

binary=$1
input=$2
output=$3
width=$4
height=$5
model=$6
log=$(mktemp)

trap 'rm -f "$log"' EXIT

{
    TIMEFORMAT='WALL %R'
    if [[ "$model" == *"/"* || "$model" == *.mxr ]]; then
        stage="upscale:model_path=${model},model_path_explicit=1,width=${width},height=${height}"
    else
        stage="upscale:model=${model},width=${width},height=${height}"
    fi
    time "$binary" \
        --input "$input" \
        --output "$output" \
        --backend migraphx \
        --preview \
        --preview-duration 60 \
        --stage "$stage"
} >"$log" 2>&1

grep -E '^(WALL|\[migraphx\])' "$log"
