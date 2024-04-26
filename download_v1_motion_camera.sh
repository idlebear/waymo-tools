#!/bin/bash

context=$1
if [ "$context" = "" ]; then
    echo "Usage: $0 <context>"
    exit 1
fi

version="1_2_1"
cache_dir="./cache/v1/motion/${version}/lidar_and_camera/training"
WaymoTrainingBucket="gs://waymo_open_dataset_motion_v_${version}/uncompressed/lidar_and_camera/training"

mkdir -p $cache_dir
gsutil -m cp $WaymoTrainingBucket/$context.tfrecord $cache_dir/$context.tfrecord


