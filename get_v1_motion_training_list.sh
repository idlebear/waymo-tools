#!/bin/bash

version="1_2_1"
WaymoLnCTrainingBucket="gs://waymo_open_dataset_motion_v_${version}/uncompressed/lidar_and_camera/training"
WaymoScenarioTrainingBucket="gs://waymo_open_dataset_motion_v_${version}/uncompressed/scenario/training"
WaymoTfExampleTrainingBucket="gs://waymo_open_dataset_motion_v_${version}/uncompressed/tf_example/training"

gsutil ls $WaymoLnCTrainingBucket | sed -n -e 's|.*/\(.*\)|\1|p' > v${version}_LnC_training_list.txt
gsutil ls $WaymoScenarioTrainingBucket | sed -n -e 's|.*/\(.*\)|\1|p' > v${version}_scenario_training_list.txt
gsutil ls $WaymoTfExampleTrainingBucket | sed -n -e 's|.*/\(.*\)|\1|p' > v${version}_tfexample_training_list.txt


