#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

TOOLS=/work/04018/wxie/maverick/nips2015-action-conditional-video-prediction/caffe/build/tools

PWD=$(pwd)
LABEL_LOC=$PWD/dummy_label.txt

DATASET_LOC=${1}

echo "Converting action keys to from 0..."
python convert_action_keys.py $DATASET_LOC

echo "Creating dummy label..."
python create_dummy_label_txt.py $DATASET_LOC

echo "Formatting image file names..."
python format_image_file_names.py $DATASET_LOC

echo "Converting to 160x210..."
for name in $DATASET_LOC*.png; do
    convert -resize 160x210\! $name $name
done


# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
# RESIZE=false
# if $RESIZE; then
#   RESIZE_HEIGHT=256
#   RESIZE_WIDTH=256
# else
#   RESIZE_HEIGHT=0
#   RESIZE_WIDTH=0
# fi
# 
# echo "Creating lmdb..."
# echo $PWD
# GLOG_logtostderr=1 $TOOLS/convert_imageset \
#     --resize_height=$RESIZE_HEIGHT \
#     --resize_width=$RESIZE_WIDTH \
#     / \
#     $LABEL_LOC \
#     $PWD/lmdb
# 
# echo "Computing mean..."
# $TOOLS/compute_image_mean ./lmdb \
#    ./mean.binaryproto
# 
# echo "Deleting lmdb..."
# rm -r lmdb

echo "Done"
