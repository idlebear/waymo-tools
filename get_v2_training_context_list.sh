#!/bin/bash

version="2_0_1"
WaymoTrainingBucket="gs://waymo_open_dataset_v_$version/training"
WaymoTag="lidar"

gsutil ls $WaymoTrainingBucket/$WaymoTag | sed -n -e 's/.*\/\([0-9_]\{1,\}\)\.parquet/\1/p' > v${version}_context_list.txt


