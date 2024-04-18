#!/bin/bash

WaymoLnCTrainingBucket="gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/lidar_and_camera/training"
Waymo20sScenarioTrainingBucket="gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario/training_20s"
WaymoScenarioTrainingBucket="gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario/training"
WaymoExampleTrainingBucket="gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/training"


# gsutil ls $WaymoLnCTrainingBucket | sed -n -e 's|.*/\([0-9a-f]\+\)\.tfrecord|\1|p' > v1_lidar_and_camera_list.txt
# gsutil ls $Waymo20sScenarioTrainingBucket | sed -n -e 's|.*/\(.*\)|\1|p' > v1_20s_training_list.txt
# gsutil ls $WaymoScenarioTrainingBucket | sed -n -e 's|.*/\(.*\)|\1|p' > v1_training_list.txt
gsutil ls $WaymoExampleTrainingBucket | sed -n -e 's|.*/\(.*\)|\1|p' > v1_example_training_list.txt


