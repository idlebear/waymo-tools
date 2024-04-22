#!/bin/bash

version="1_4_3"
WaymoExampleTrainingBucket="gs://waymo_open_dataset_v_${version}/individual_files/training"

gsutil ls $WaymoExampleTrainingBucket | sed -n -e 's|.*/\(.*\)|\1|p' > v${version}_training_list.txt


