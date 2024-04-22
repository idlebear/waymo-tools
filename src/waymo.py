#
# Experiments with loading data from the waymo open dataset
#
from typing import Optional
import warnings

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import os
import itertools
import math
import random

import tensorrt
import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.v2.perception.utils import lidar_utils

import open3d as o3d
import time

# Path to the directory with all components

# BUGBUG -- playing with the testing data because it's much smaller than the full training set
dataset_dir = "./test/v2/2_0_1/training"


def load_context_names(dataset_dir: str, tag: str) -> list:
    context_paths = os.path.join(dataset_dir, tag)
    contexts = os.listdir(context_paths)
    contexts = [os.path.splitext(context)[0] for context in contexts if context.startswith("_metadata") is False]
    return contexts


def read(dataset_dir: str, context_str: str, tag: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = tf.io.gfile.glob(f"{dataset_dir}/{tag}/{context_str}.parquet")
    return dd.read_parquet(paths)


# Borrowing some sample code to display the point cloud
#
#  https://salzi.blog/2022/05/14/waymo-open-dataset-open3d-point-cloud-viewer/
#
def show_point_cloud(points: np.ndarray) -> None:
    # pylint: disable=no-member (E1101)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.voxel_down_sample(voxel_size=0.1)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.6, origin=[0, 0, 0])

    pcd.points = o3d.utility.Vector3dVector(points)

    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)

    # for label in laser_labels:
    #     bbox_corners = transform_bbox_waymo(label)
    #     bbox_points = build_open3d_bbox(bbox_corners, label)

    #     colors = [[1, 0, 0] for _ in range(len(LINE_SEGMENTS))]
    #     line_set = o3d.geometry.LineSet(
    #         points=o3d.utility.Vector3dVector(bbox_points),
    #         lines=o3d.utility.Vector2iVector(LINE_SEGMENTS),
    #     )

    #     line_set.colors = o3d.utility.Vector3dVector(colors)
    #     vis.add_geometry(line_set)

    vis.run()


tag_list = os.listdir(dataset_dir)
context_name_list = load_context_names(dataset_dir, tag_list[0])

for context in context_name_list:

    component_load_start = time.time()

    # select only the rows from the lidar data that correspond to the medium range lidar
    lidar_df = read(dataset_dir=dataset_dir, context_str=context, tag="lidar")
    lidar_df = lidar_df.loc[lidar_df["key.laser_name"] == 1]
    lidar_iter = iter(lidar_df.iterrows())

    lidar_calibration_df = read(dataset_dir=dataset_dir, context_str=context, tag="lidar_calibration")

    lidar_pose_df = read(dataset_dir=dataset_dir, context_str=context, tag="lidar_pose")
    lidar_pose_df = lidar_pose_df.loc[lidar_pose_df["key.laser_name"] == 1]
    lidar_pose_iter = iter(lidar_pose_df.iterrows())

    vehicle_pose_df = read(dataset_dir=dataset_dir, context_str=context, tag="vehicle_pose")
    vehicle_pose_iter = iter(vehicle_pose_df.iterrows())

    # get the lidar calibration data for the meadium range lidar
    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(
        lidar_calibration_df.loc[lidar_calibration_df["key.laser_name"] == 1].compute().iloc[0]
    )

    stats_df = read(dataset_dir=dataset_dir, context_str=context, tag="stats")


    component_load_end = time.time()

    start_time = None
    last_time = None

    print(f"Component load time: {component_load_end - component_load_start}")

    # interate of the rows of the lidar data
    while True:

        try:
            load_start = time.time()

            lidar = v2.LiDARComponent.from_dict(next(lidar_iter)[1])
            lidar_pose = v2.LiDARPoseComponent.from_dict(next(lidar_pose_iter)[1])

            if lidar.key.frame_timestamp_micros != lidar_pose.key.frame_timestamp_micros:
                print(
                    f"ERROR: Frame timestamp: {lidar.key.frame_timestamp_micros}, Pose timestamp: {lidar_pose.key.frame_timestamp_micros}"
                )
                continue

            vehicle_pose = v2.VehiclePoseComponent.from_dict(next(vehicle_pose_iter)[1])
            load_end = time.time()
            print(f"Load time: {load_end - load_start}")

            point_conversion_start = time.time()
            points = lidar_utils.convert_range_image_to_point_cloud(
                lidar.range_image_return1, lidar_calibration, lidar_pose.range_image_return1, vehicle_pose, False
            )
            point_conversion_end = time.time()
            print(f"Point conversion time: {point_conversion_end - point_conversion_start}")

            if start_time is None:
                start_time = lidar.key.frame_timestamp_micros
                dt = 0
            else:
                dt = lidar.key.frame_timestamp_micros - last_time
            last_time = lidar.key.frame_timestamp_micros

            show_point_cloud(points)
            print(f"Time between frames (uS): {dt}")

        except StopIteration:
            break
