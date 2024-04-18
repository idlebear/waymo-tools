#!/bin/bash

WaymoTrainingBucket="gs://waymo_open_dataset_v_2_0_1/training"
WaymoTag="lidar"

gsutil ls $WaymoTrainingBucket/$WaymoTag | sed -n -e 's/.*\/\([0-9_]\{1,\}\)\.parquet/\1/p' > v2_context_list.txt


