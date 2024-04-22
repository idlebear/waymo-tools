#!/bin/bash

context=$1
if [ "$context" = "" ]; then
    echo "Usage: $0 <context>"
    exit 1
fi

version="1_4_3"
cache_dir="./test/v1/perception/${version}/training"
WaymoTrainingBucket="gs://waymo_open_dataset_v_${version}/individual_files/training"

mkdir -p $cache_dir
gsutil -m cp ${WaymoTrainingBucket}/$context $cache_dir/$context


