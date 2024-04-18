#!/bin/bash

context=$1
if [ "$context" = "" ]; then
    echo "Usage: $0 <context>"
    exit 1
fi

cache_dir="./test/v1"
WaymoTrainingBucket="gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario/training_20s"

mkdir -p $cache_dir/scenario/training_20s/$context
gsutil -m cp ${WaymoTrainingBucket}/$context $cache_dir/scenario/training_20s/$context


