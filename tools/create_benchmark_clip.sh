#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
out_dir="$repo_root/benchmarks"
out_file="$out_dir/ave_benchmark_960x540_20s.mp4"

mkdir -p "$out_dir"

font_file=""
if command -v fc-match >/dev/null 2>&1; then
    font_file=$(fc-match -f '%{file}\n' 'DejaVu Sans:style=Bold' | head -n 1 || true)
fi

drawtext_filter=""
if [[ -n "$font_file" ]]; then
    drawtext_filter=",drawtext=fontfile=${font_file}:text='AVE BENCH | %{pts\\:hms} | OpenProteus x2 / ClearReality x4':x=28:y=28:fontsize=30:fontcolor=white:borderw=2:bordercolor=black@0.90"
fi

ffmpeg -y \
    -f lavfi -i "testsrc2=size=960x540:rate=30:duration=20" \
    -vf "format=yuv420p,drawgrid=width=64:height=64:thickness=1:color=white@0.18,drawbox=x='(w-220)/2 + (w-220)/4*sin(2*PI*t/6)':y='h*0.18':w=220:h=120:color=black@0.32:t=fill,drawbox=x='40 + 120*sin(2*PI*t/5)':y='h-180':w=180:h=90:color=yellow@0.25:t=fill${drawtext_filter}" \
    -t 20 \
    -c:v libx264 \
    -preset medium \
    -crf 18 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "$out_file"

ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames \
    -show_entries format=duration \
    -of default=noprint_wrappers=1 \
    "$out_file"

echo "$out_file"
