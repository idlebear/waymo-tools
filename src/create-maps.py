import json
import pickle
import tensorrt  # import before tensorflow to prevent TensorRT not found error
import tensorflow as tf
from typing import List
import tqdm

import numpy as np
import argparse
import os

from PIL import Image
import matplotlib.pyplot as plt

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps
from waymo_open_dataset.utils import transform_utils

from dogm_py import LaserMeasurementGridParams
from dogm_py import LaserMeasurementGrid
from dogm_py import DOGMParams
from dogm_py import DOGM
from dogm_py import VectorFloat
from dogm_py import renderOccupancyGrid
from dogm_py import renderMeasurement

from Grid.GridMap import ProbabilityGrid

FIG_LIDAR_MAP = 1
FIG_OCCUPANCY_GRID = 2
FIG_DYNAMIC_OCCUPANCY_GRID = 3

DEFAULT_CACHE_LOC = "./cache"
CACHE_PATH = "v1/perception/1_4_3/training"
MAP_PATH = "v1/maps"


LIDAR_RANGE = 75.0
LIDAR_RAYS = 2650.0
LIDAR_INCREMENT = (np.pi * 2.0) / LIDAR_RAYS

GRID_WIDTH = 2 * LIDAR_RANGE
GRID_CELL_WIDTH = 0.5
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


def update_occupancy_grid(occupancy_grid, measurements, pose_data):

    if occupancy_grid is None:
        occupancy_grid = ProbabilityGrid(pose_data[:2], 500, 500, resolution=GRID_CELL_WIDTH)

    # convert the measurements to probabilities
    prob_occ = measurements[0, ...] + 0.5 * (1.0 - measurements[0, ...] - measurements[1, ...])

    occupancy_grid.set_probability_at(
        (
            pose_data[0] - GRID_WIDTH / 2.0,
            pose_data[1] - GRID_WIDTH / 2.0,
        ),
        prob_occ,
    )

    return occupancy_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Waymo Open Dataset")
    parser.add_argument("--cache_location", type=str, default=DEFAULT_CACHE_LOC, help="Cache location")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    return parser.parse_args()


def convert_range_image_to_2D(frame, range_images, range_image_index, range_image_top_poses, lidar_calibrations):
    """Convert the range images int bucketized lidar rays."""

    calibrations = sorted(lidar_calibrations, key=lambda c: c.name)

    num_buckets = int(np.pi * 2.0 / LIDAR_INCREMENT) + 1
    # initialize a tensor to hold the ranges
    # ranges = np.ones(num_buckets) * np.inf
    # # Create an n x 1 tensor of inf values
    # ranges = tf.fill([num_buckets], float("inf"))
    ranges = tf.fill([num_buckets], float(LIDAR_RANGE + 1))

    cartesian_points = frame_utils.convert_range_image_to_cartesian(
        frame, range_images, range_image_top_poses, range_image_index, keep_polar_features=False
    )

    raw_points = None

    for calibration in calibrations:

        if calibration.name != 1:
            continue

        # Filter out any points at or below the ground plane
        points = cartesian_points[calibration.name]

        # Create a mask for the height range
        points = tf.boolean_mask(points, (points[..., 2] > Z_MIN) & (points[..., 2] < Z_MAX))
        points = tf.reshape(points, [-1, 3])

        if raw_points is None:
            raw_points = points
        else:
            raw_points = tf.concat([raw_points, points], axis=0)

        # Convert the points to lidar rays
        # use tf to find the ranges (ignore the z component of the points)
        point_ranges = tf.math.reduce_euclidean_norm(points[:, :2], axis=1)
        point_angles = (
            tf.clip_by_value((tf.atan2(points[:, 1], points[:, 0]) + np.pi), 0, np.pi * 2.0) / LIDAR_INCREMENT
        )

        # if calibration.name != 1:
        #     valid_range_mask = tf.math.logical_and(point_ranges > 10, point_ranges < LIDAR_RANGE)
        #     point_ranges = tf.boolean_mask(point_ranges, valid_range_mask)
        #     point_angles = tf.boolean_mask(point_angles, valid_range_mask)

        point_angles = tf.cast(point_angles, tf.int32)

        # Get the indices that would sort the angles
        sorted_indices = tf.argsort(point_angles)

        # Use these indices to sort the angles and the ranges
        sorted_angles = tf.gather(point_angles, sorted_indices)
        sorted_ranges = tf.gather(point_ranges, sorted_indices)

        # Now you can find the minimum range for each angle
        indices = tf.unique(sorted_angles).y
        updated_ranges = tf.gather(tf.math.segment_min(sorted_ranges, sorted_angles), indices)

        # # update the ranges
        # for index, min_range in zip(indices, min_ranges):
        #     if min_range < ranges[index]:
        #         ranges[index] = min_range
        current_ranges = tf.gather(ranges, indices)
        updates = tf.where(current_ranges > updated_ranges, updated_ranges, current_ranges)
        ranges = tf.tensor_scatter_nd_update(ranges, tf.expand_dims(indices, axis=-1), updates)

    # return ranges, raw_points
    return ranges.numpy(), raw_points


def get_contexts(cache_location: str):
    """Get the contexts from the cache location.

    Args:
      cache_location: The location of the cache.

    Returns:
      A list of context strings.
    """

    contexts = []
    for filename in tf.io.gfile.glob(os.path.join(cache_location, "*.tfrecord")):
        context = os.path.basename(filename)
        contexts.append(context)

    return contexts


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
        filename = os.path.join(cache_location, f"{context}.tfrecord")
        if not tf.io.gfile.exists(filename):
            print(f"Context {context} not found in cache.")
            filename = None

    if filename is None:
        print("Selecting a random context.")
        filenames = tf.io.gfile.glob(os.path.join(cache_location, "*.tfrecord"))
        filename = rng.choice(filenames)
        # update the context by taking the base name and extension from the path
        context = os.path.basename(filename)

    # Load the dataset from the cache.
    dataset = tf.data.TFRecordDataset(filename, compression_type="")

    # return both the dataset and the context
    return dataset, context


def construct_laser_measurement_grid():

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

    return lmg


def main():
    # parse arguments
    args = parse_args()

    # create a generator
    rng = np.random.default_rng(seed=args.seed)

    # get the contexts from the cache
    cache = os.path.join(args.cache_location, CACHE_PATH)
    contexts = get_contexts(cache)

    for context in tqdm.tqdm(contexts):
        # load the dataset
        dataset, context = load_dataset(cache, context, rng)

        lmg = construct_laser_measurement_grid()

        occupancy_grid = None

        nominal_trajectory = []

        t = 0
        last_time = None

        for data in tqdm.tqdm(dataset):

            frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))

            range_images, camera_projections, seg_labels, range_image_top_poses = (
                frame_utils.parse_range_image_and_camera_projection(frame)
            )

            # convert the range image to ranges
            ranges, points = convert_range_image_to_2D(
                frame=frame,
                range_images=range_images,
                range_image_index=0,
                range_image_top_poses=range_image_top_poses,
                lidar_calibrations=frame.context.laser_calibrations,
            )
            ranges = ranges.astype(np.float32)

            # # calculate the current yaw
            frame_transform = np.reshape(np.array(frame.pose.transform), [4, 4])
            rotated = np.dot(frame_transform, np.array([1, 0, 0, 0]))
            yaw = np.arctan2(rotated[1], rotated[0]).astype(np.float32)

            dt = frame.timestamp_micros - last_time if last_time is not None else 0
            t += dt
            last_time = frame.timestamp_micros
            nominal_trajectory.append([t, dt, float(frame_transform[0, 3]), float(frame_transform[1, 3]), float(yaw)])

            # update the occupancy grid
            grid_data = lmg.generateGrid(VectorFloat(ranges), np.rad2deg(yaw))

            measurements = renderMeasurement(grid_data).reshape(2, GRID_SIZE, GRID_SIZE)
            occupancy_grid = update_occupancy_grid(
                occupancy_grid=occupancy_grid,
                measurements=measurements,
                pose_data=(frame_transform[0, 3], frame_transform[1, 3], yaw),
            )

        # check if the map directory exists and create it if it does not
        map_path = f"{args.cache_location}/{MAP_PATH}"
        if not os.path.exists(map_path):
            os.makedirs(map_path)

        # write the map data to a json file
        occupancy_grid_filename = f"{context}.pkl"
        map_data = {
            "context": context,
            "origin": list(occupancy_grid._origin),
            "height": occupancy_grid._height,
            "width": occupancy_grid._width,
            "resolution": occupancy_grid._resolution,
            "trajectory": nominal_trajectory,
            "map": occupancy_grid_filename,
        }

        with open(f"{map_path}/{occupancy_grid_filename}", "wb") as fp:
            pickle.dump(occupancy_grid, fp)
        with open(f"{map_path}/{context}.json", "w") as fp:
            json.dump(map_data, fp)
        map_img = Image.fromarray((occupancy_grid.probability_map() * 255).astype(np.uint8)).convert("RGB")
        map_img.save(f"{map_path}/{context}.png")


if __name__ == "__main__":
    main()
