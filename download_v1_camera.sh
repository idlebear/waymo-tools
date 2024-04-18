#!/bin/bash

context=$1
if [ "$context" = "" ]; then
    echo "Usage: $0 <context>"
    exit 1
fi

cache_dir="./test/v1"
WaymoTrainingBucket="gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/lidar_and_camera/training"

mkdir -p $cache_dir/lidar_and_camera/training
gsutil -m cp $WaymoTrainingBucket/$context.tfrecord $cache_dir/lidar_and_camera/training/$context.tfrecord


