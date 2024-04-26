#
# Experiments with loading data from the waymo open dataset
#
from typing import Optional
import warnings

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import os
import argparse
import os

from PIL import Image
import matplotlib.pyplot as plt

import open3d as o3d
import time

import tensorrt  # import before tensorflow to prevent TensorRT not found error
import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2

from waymo_open_dataset.v2.perception.utils import lidar_utils


from dogm_py import LaserMeasurementGridParams
from dogm_py import LaserMeasurementGrid
from dogm_py import DOGMParams
from dogm_py import DOGM
from dogm_py import VectorFloat
from dogm_py import renderOccupancyGrid
from dogm_py import renderMeasurement


# Path to the directory with all components


FIG_LIDAR_MAP = 1
FIG_OCCUPANCY_GRID = 2

DEFAULT_CACHE = "./cache/v2/perception/2_0_1/training"

LIDAR_RANGE = 75.0
LIDAR_RAYS = 2650.0
LIDAR_INCREMENT = (np.pi * 2.0) / LIDAR_RAYS

GRID_WIDTH = 2 * LIDAR_RANGE
GRID_CELL_WIDTH = 0.25
GRID_SIZE = int(GRID_WIDTH / GRID_CELL_WIDTH)

LIDAR_LOWER_X_BOUND = -GRID_WIDTH / 2
LIDAR_LOWER_Y_BOUND = -GRID_WIDTH / 2
LIDAR_LOWER_Z_BOUND = 0.0
LIDAR_UPPER_X_BOUND = GRID_WIDTH / 2
LIDAR_UPPER_Y_BOUND = GRID_WIDTH / 2
LIDAR_UPPER_Z_BOUND = 5.0

Z_MIN = 0.5
Z_MAX = 2.0

# the voxel map is scaled at 0.2 m/pixel -- use that value to scale the
# voxel cube
VOXEL_MAP_SCALE = 0.1
VOXEL_MAP_X = int((LIDAR_UPPER_X_BOUND - LIDAR_LOWER_X_BOUND) / VOXEL_MAP_SCALE)
VOXEL_MAP_Y = int((LIDAR_UPPER_Y_BOUND - LIDAR_LOWER_Y_BOUND) / VOXEL_MAP_SCALE)
VOXEL_MAP_Z = int((LIDAR_UPPER_Z_BOUND - LIDAR_LOWER_Z_BOUND) / VOXEL_MAP_SCALE)

OCCUPANCY_GRID_X = int((LIDAR_UPPER_X_BOUND - LIDAR_LOWER_X_BOUND) / GRID_CELL_WIDTH)
OCCUPANCY_GRID_Y = int((LIDAR_UPPER_Y_BOUND - LIDAR_LOWER_Y_BOUND) / GRID_CELL_WIDTH)


# desired tags from dataset
data_tags = ["lidar", "lidar_calibration", "lidar_pose", "lidar_box", "vehicle_pose", "stats"]


def construct_occupancy_grid():
    # DOGM params
    particle_count = 20000
    new_born_particle_count = 10000
    persistance_prob = 0.5
    stddev_process_noise_position = 0.1
    stddev_process_noise_velocity = 0.1
    birth_prob = 0.2
    stddev_velocity = 1.0
    init_max_velocity = 5.0
    dogm_params = DOGMParams(
        size=GRID_WIDTH,
        resolution=GRID_CELL_WIDTH,
        particle_count=particle_count,
        new_born_particle_count=new_born_particle_count,
        persistance_prob=persistance_prob,
        stddev_process_noise_position=stddev_process_noise_position,
        stddev_process_noise_velocity=stddev_process_noise_velocity,
        birth_prob=birth_prob,
        stddev_velocity=stddev_velocity,
        init_max_velocity=init_max_velocity,
    )
    dogm = DOGM(params=dogm_params)

    # Create a LaserMeasurementGrid object that converts the range based laserscan update into a cartesian
    # grid.  The cartesian grid is then used as an update to the occupancy grid.
    fov = 360.0
    lidar_range = LIDAR_RANGE
    lidar_res = VOXEL_MAP_SCALE
    stddev_range = 0.1
    lmg_params = LaserMeasurementGridParams(
        fov=fov,
        angle_increment=LIDAR_INCREMENT * 180.0 / np.pi,
        max_range=lidar_range,
        resolution=lidar_res,
        stddev_range=stddev_range,
    )
    lmg = LaserMeasurementGrid(params=lmg_params, size=GRID_WIDTH, resolution=GRID_CELL_WIDTH)

    return dogm, lmg


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Waymo Open Dataset")
    parser.add_argument("--cache_location", type=str, default=DEFAULT_CACHE, help="Cache location")
    parser.add_argument(
        "--context", type=str, default=None, help="Context string.  If not provided, a random context is selected."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    return parser.parse_args()


def convert_range_image_to_2D(frame, range_images, range_image_index, range_image_top_poses, lidar_calibrations):
    """Convert the range images int bucketized lidar rays."""


def load_context_names(dataset_dir: str, tag: str) -> list:
    context_paths = os.path.join(dataset_dir, tag)
    contexts = os.listdir(context_paths)
    contexts = [os.path.splitext(context)[0] for context in contexts if context.startswith("_metadata") is False]
    return contexts


def read(dataset_dir: str, context_str: str, tag: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = tf.io.gfile.glob(f"{dataset_dir}/{tag}/{context_str}.parquet")
    return dd.read_parquet(paths)


def convert_point_cloud_to_2D(points, vehicle_pose):
    """Convert the range images int bucketized lidar rays."""

    num_buckets = int(np.pi * 2.0 / LIDAR_INCREMENT) + 1
    # initialize a tensor to hold the ranges
    # ranges = np.ones(num_buckets) * np.inf
    # # Create an n x 1 tensor of inf values
    # ranges = tf.fill([num_buckets], float("inf"))
    ranges = tf.fill([num_buckets], float(LIDAR_RANGE + 1))

    # Create a mask for the height range
    points = tf.boolean_mask(points, (points[..., 2] > Z_MIN) & (points[..., 2] < Z_MAX))
    points = tf.reshape(points, [-1, 3])

    # Convert the points to lidar rays
    # use tf to find the ranges (ignore the z component of the points)
    point_ranges = tf.math.reduce_euclidean_norm(points[:, :2], axis=1)
    point_angles = tf.clip_by_value((tf.atan2(points[:, 1], points[:, 0]) + np.pi), 0, np.pi * 2.0) / LIDAR_INCREMENT
    point_angles = tf.cast(point_angles, tf.int32)

    # Get the indices that would sort the angles
    sorted_indices = tf.argsort(point_angles)

    # Use these indices to sort the angles and the ranges
    sorted_angles = tf.gather(point_angles, sorted_indices)
    sorted_ranges = tf.gather(point_ranges, sorted_indices)

    # Now you can find the minimum range for each angle
    indices = tf.unique(sorted_angles).y
    updated_ranges = tf.gather(tf.math.segment_min(sorted_ranges, sorted_angles), indices)

    # update the ranges
    current_ranges = tf.gather(ranges, indices)
    updates = tf.where(current_ranges > updated_ranges, updated_ranges, current_ranges)
    ranges = tf.tensor_scatter_nd_update(ranges, tf.expand_dims(indices, axis=-1), updates)

    # return ranges, raw_points
    return ranges.numpy()


def load_dataset(cache_location: str, context: str, rng: np.random.Generator):
    """Load the Waymo Open Dataset from the given cache location.

    Args:
      cache_location: The location of the dataset cache.
      context: The context string to filter the dataset.
      rng: The random number generator.

    Returns:
      A tf.data.Dataset containing the Waymo Open Dataset.
    """

    filename = None
    if context is not None:
        print(f"Filtering dataset by context: {context}")
    else:
        print("Selecting a random context.")
        context_name_list = load_context_names(cache_location, data_tags[0])
        context = rng.choice(context_name_list)

    # Load the dataset from the cache.
    dataset = {}
    for tag in data_tags:
        df = read(dataset_dir=cache_location, context_str=context, tag=tag)
        dataset[tag] = df

    return dataset


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


def main():
    # parse arguments
    args = parse_args()

    # create a generator
    rng = np.random.default_rng(seed=args.seed)

    # set up some display parameters
    map_fig, map_ax = plt.subplots(1, 1, num=FIG_OCCUPANCY_GRID, figsize=(10, 10))
    map_im = map_ax.imshow(np.zeros([GRID_SIZE, GRID_SIZE, 3], dtype=np.uint8))

    # bev_fig, bev_ax = plt.subplots(1, 1, num=FIG_OCCUPANCY_GRID, figsize=(10, 10))
    # bev_im = bev_ax.imshow(np.zeros([1000, 1000, 3], dtype=np.uint8))

    plt.show(block=False)

    # construct the occupancy grid
    dogm, lmg = construct_occupancy_grid()

    # load the dataset
    dataset = load_dataset(args.cache_location, args.context, rng)

    lidar_df = v2.merge(dataset["lidar"], dataset["lidar_pose"])
    lidar_box_df = dataset["lidar_box"]
    lidar_calibration_df = dataset["lidar_calibration"]
    lidar_df = lidar_df.loc[lidar_df["key.laser_name"] == 1]
    lidar_iterator = iter(lidar_df.iterrows())

    vehicle_pose_df = dataset["vehicle_pose"]
    vehicle_pose_iter = iter(vehicle_pose_df.iterrows())

    # lidar_names = sorted(list(lidar_calibration_df["key.laser_name"].unique()))
    lidar_name = 1
    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(
        lidar_calibration_df.loc[lidar_calibration_df["key.laser_name"] == lidar_name].compute().iloc[0]
    )

    # lidar_calibrations = {}
    # lidar_views = {}
    # lidar_iterators = {}
    # lidar_pose_views = {}
    # lidar_pose_iterators = {}

    # for lidar_name in lidar_names:
    #     lidar_calibrations[lidar_name] = v2.LiDARCalibrationComponent.from_dict(
    #         lidar_calibration_df.loc[lidar_calibration_df["key.laser_name"] == lidar_name].compute().iloc[0]
    #     )
    #     lidar_views[lidar_name] = lidar_df.loc[lidar_df["key.laser_name"] == lidar_name]
    #     lidar_iterators[lidar_name] = iter(lidar_views[lidar_name].iterrows())

    start_time = None
    last_time = None

    # interate of the rows of the lidar data
    while True:

        try:
            load_start = time.time()

            vehicle_pose = v2.VehiclePoseComponent.from_dict(next(vehicle_pose_iter)[1])
            load_end = time.time()
            print(f"Load time: {load_end - load_start}")

            # dump the timestamp difference
            if start_time is None:
                start_time = vehicle_pose.key.frame_timestamp_micros
                dt = 0
            else:
                dt = vehicle_pose.key.frame_timestamp_micros - last_time
            last_time = vehicle_pose.key.frame_timestamp_micros
            print(f"Time between frames (uS): {dt}")

            points = None

            _, row = next(lidar_iterator)
            lidar = v2.LiDARComponent.from_dict(row)
            lidar_pose = v2.LiDARPoseComponent.from_dict(row)

            lidar_points = lidar_utils.convert_range_image_to_point_cloud(
                lidar.range_image_return1,
                lidar_calibration,
                lidar_pose.range_image_return1,
                vehicle_pose,
                False,
            )

            # convert the point cloud to 2D
            ranges = convert_point_cloud_to_2D(lidar_points, vehicle_pose)
            ranges = ranges.astype(np.float32)

            fig = plt.figure(FIG_LIDAR_MAP)
            plt.clf()
            plt.scatter(lidar_points[:, 0], lidar_points[:, 1], s=0.5)
            plt.xlim([-100, 100])
            plt.ylim([-100, 100])

            # calculate the current yaw - there is probably a less convoluted way to do this
            frame_transform = vehicle_pose.world_from_vehicle.transform.reshape(4, 4)
            rotated = np.dot(frame_transform, np.array([1, 0, 0, 0]))
            yaw = np.arctan2(rotated[1], rotated[0]).astype(np.float32)

            # update the occupancy grid
            grid_data = lmg.generateGrid(VectorFloat(ranges), np.rad2deg(yaw))
            dogm.updateGrid(grid_data, frame_transform[0, 3], frame_transform[1, 3], 0.1)

            # render the occupancy grid
            current_occ_grid = renderOccupancyGrid(dogm)

            map_img = Image.fromarray(((1 - current_occ_grid) * 255).astype(np.uint8)).convert("RGB")
            map_im.set_data(map_img)

            plt.pause(1)

        except StopIteration:
            break


if __name__ == "__main__":
    main()
