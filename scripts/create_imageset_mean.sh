#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

DATA=data/ilsvrc12
TOOLS=../caffe/build/tools

$TOOLS/compute_image_mean ../pong_lmdb \
  ./mean.binaryproto

echo "Done."
