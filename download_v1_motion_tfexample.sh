#!/bin/bash

context=$1
if [ "$context" = "" ]; then
    echo "Usage: $0 <context>"
    exit 1
fi

version="1_2_1"
cache_dir="./test/v1/motion/${version}/tf_example/training"
WaymoTrainingBucket="gs://waymo_open_dataset_motion_v_${version}/uncompressed/tf_example/training"

mkdir -p $cache_dir
gsutil -m cp ${WaymoTrainingBucket}/$context $cache_dir/$context


