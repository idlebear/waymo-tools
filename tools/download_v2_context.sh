#!/bin/bash

context=$1
if [ "$context" = "" ]; then
    echo "Usage: $0 <context>"
    exit 1
fi

cache_dir="./cache/v2"
version="2_0_1"
WaymoTrainingBucket="gs://waymo_open_dataset_v_$version/training"

# tag_list = (
#     "camera_box"
#     "camera_calibration"
#     "camera_hkp"
#     "camera_image"
#     "camera_segmentation"
#     "camera_to_lidar_box_association"
#     "lidar"
#     "lidar_box"
#     "lidar_camera_synced_box"
#     "lidar_calibration"
#     "lidar_pose"
#     "lidar_hkp"
#     "lidar_camera_projection"
#     "lidar_segmentation"
#     "projected_lidar_box"
#     "stats"
#     "vehicle_pose"
# )


tags=(
     "camera_box"
     "camera_calibration"
     "camera_hkp"
     "camera_image"
     "camera_segmentation"
     "camera_to_lidar_box_association"
     "lidar"
     "lidar_box"
     "lidar_camera_synced_box"
     "lidar_calibration"
     "lidar_pose"
     "lidar_hkp"
     "lidar_camera_projection"
     "lidar_segmentation"
     "projected_lidar_box"
     "stats"
     "vehicle_pose"
    )

for tag in "${tags[@]}"
do
    mkdir -p $cache_dir/perception/$version/training/$tag
    gsutil -m cp gs://waymo_open_dataset_v_$version/training/$tag/$context.parquet $cache_dir/perception/$version/training/$tag/$context.parquet
done


