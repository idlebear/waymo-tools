import tensorrt
from typing import List

import numpy as np
import plotly.graph_objs as go
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import scenario_pb2

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps


def plot_point_clouds_with_maps(frames: List[dataset_pb2.Frame]) -> None:
    """Plot the point clouds within the given frames with map data.

    Map data must be populated in the first frame in the list.

    Args:
      frames: A list of frames to be plotted, frames[0] must contain map data.
    """

    # Plot the map features.
    if len(frames) == 0:
        return
    figure = plot_maps.plot_map_features(frames[0].map_features)

    for frame in frames:
        # Parse the frame lidar data into range images.
        range_images, camera_projections, seg_labels, range_image_top_poses = (
            frame_utils.parse_range_image_and_camera_projection(frame)
        )

        # Project the range images into points.
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_poses,
            keep_polar_features=True,
        )
        xyz = points[0][:, 3:]
        num_points = xyz.shape[0]

        # Transform the points from the vehicle frame to the world frame.
        xyz = np.concatenate([xyz, np.ones([num_points, 1])], axis=-1)
        transform = np.reshape(np.array(frame.pose.transform), [4, 4])
        xyz = np.transpose(np.matmul(transform, np.transpose(xyz)))[:, 0:3]

        # Correct the pose of the points into the coordinate system of the first
        # frame to align with the map data.
        offset = frame.map_pose_offset
        points_offset = np.array([offset.x, offset.y, offset.z])
        xyz += points_offset

        # Plot the point cloud for this frame aligned with the map data.
        intensity = points[0][:, 0]
        figure.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(
                    size=1,
                    color=intensity,  # set color to an array/list of desired values
                    colorscale="Pinkyl",  # choose a colorscale
                    opacity=0.8,
                ),
            )
        )

    figure.show()


import tensorflow as tf

# FILENAME = "./test/v1/scenario/training/training.tfrecord-00006-of-01000"
# FILENAME = "./test/v1/scenario/training_20s/training_20s.tfrecord-00029-of-01000"
FILENAME = "./test/v1/lidar_and_camera/training/100006d9c3e93b6e.tfrecord"

# See https://www.tensorflow.org/tutorials/load_data/tfrecord
dataset = tf.data.TFRecordDataset(FILENAME, compression_type="")

# Load only 2 frames. Note that using too many frames may be slow to display.
frames = []
count = 0
for index, data in enumerate(dataset):

    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    # proto = scenario_pb2.Scenario()
    # proto.ParseFromString(bytearray(data.numpy()))

    # frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
    frames.append(frame)
    count += 1
    if count == 2:
        break


# Interactive plot of multiple point clouds aligned to the maps frame.

# For most systems:
#   left mouse button:   rotate
#   right mouse button:  pan
#   scroll wheel:        zoom

plot_point_clouds_with_maps(frames)
