#!/bin/bash

cache_loc=$1
version=$2
bucket=$3
context=$4

if [ "$context" = "" ]; then
    echo "Usage: $0 <cache loc> <version> <bucket> <context>"
    exit 1
fi

cache_dir="${cache_loc}/v1/perception/${version}/${bucket}"
WaymoBucket="gs://waymo_open_dataset_v_${version}/individual_files/${bucket}"

mkdir -p $cache_dir
gsutil -m cp ${WaymoBucket}/$context $cache_dir/$context


