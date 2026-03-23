#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
binary=${1:-"$repo_root/build/ave"}
input=${2:-"$repo_root/benchmarks/ave_benchmark_960x540_20s.mp4"}
out_dir=${3:-"$repo_root/benchmarks/results"}
benchmark_env=(
    HIP_VISIBLE_DEVICES=0
    ROCR_VISIBLE_DEVICES=0
    AVE_MIGRAPHX_VISIBLE_DEVICES=0
    MIOPEN_FIND_MODE=FAST
    MIOPEN_COMPILE_PARALLEL_LEVEL=1
)

mkdir -p "$out_dir"

env "${benchmark_env[@]}" "$repo_root/tools/bench_ave.sh" \
    "$binary" \
    "$input" \
    "$out_dir/openproteus-compact-x2_1920x1080.mp4" \
    1920 \
    1080 \
    openproteus-compact-x2

env "${benchmark_env[@]}" "$repo_root/tools/bench_ave.sh" \
    "$binary" \
    "$input" \
    "$out_dir/clearreality-x4-fast_3840x2160.mp4" \
    3840 \
    2160 \
    clearreality-x4-fast
