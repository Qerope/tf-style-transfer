#!/bin/sh
# usage: stylize_video.sh clip.mp4 style.jpg
set -e
clip=$1
style=$2
mkdir -p frames out
ffmpeg -i "$clip" -qscale:v 2 frames/%05d.jpg
for f in frames/*.jpg; do
    n=$(basename "$f" .jpg)
    python style.py --content "$f" --style "$style" --out "out/$n.png" --iterations 100
done
ffmpeg -framerate 24 -i out/%05d.png -c:v libx264 -pix_fmt yuv420p stylized.mp4
