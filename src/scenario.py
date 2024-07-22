# Scenario.py -- code to load and execute a scenario

import json
import numpy as np
import os
import pickle

import tensorrt  # import before tensorflow to prevent TensorRT not found error
import tensorflow as tf
from typing import List

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps
from waymo_open_dataset.utils import transform_utils

from Grid.GridMap import ProbabilityGrid

from config import CACHE_PATH, MAP_PATH

Z_MIN = 0.75
Z_MAX = 2.5

from feature_maps import create_maps


class Scenario:
    def __init__(self, context, cache_location, scan_params, logger=None):

        self.context = context
        self.cache_location = cache_location
        self.scan_params = scan_params

        with open(f"{cache_location}/{MAP_PATH}/{context}.json", "r") as f:
            self.data = json.load(f)

        self.trajectory = self.data["trajectory"]
        self.dt = 0.1  # TODO: hardcoded, but could pull from cache: np.mean(np.diff(self._t) / 1.0e6)

        with open(f"{cache_location}/{MAP_PATH}/{self.data['map']}", "rb") as f:
            self.map = pickle.load(f)
        self.start_pos = self.trajectory[0]

        self.scenario_map_data = None
        self.map_origin = None
        self.map_pixels_per_meter = 5

        self.load_dataset()
        # load the first frame
        for data in self.dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            if frame.map_features is not None and len(frame.map_features) > 0:
                self.scenario_map_data, self.map_origin = create_maps(
                    frame.map_features, pixels_per_meter=self.map_pixels_per_meter
                )
            break

        self.reset()

    def load_dataset(self):
        filename = f"{self.cache_location}/{CACHE_PATH}/{self.context}"
        if not tf.io.gfile.exists(filename):
            raise ValueError(f"Context {self.context} not found in cache.")

        # Load the dataset from the cache.
        self.dataset = tf.data.TFRecordDataset(filename, compression_type="")

    def __frame_generator(self):
        for data in self.dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            if frame.map_features is not None and len(frame.map_features) > 0:
                self.scenario_map_data, self.map_origin = create_maps(
                    frame.map_features, pixels_per_meter=self.map_pixels_per_meter
                )

                # import matplotlib.pyplot as plt
                # for i, layer in enumerate(self.scenario_map_data):
                #     plt.figure(figsize=(10, 10), num=i)
                #     plt.imshow(layer)
                # plt.show(block=False)

            yield frame

    def reset(self):
        self.pos = self.start_pos
        self.frames = self.__frame_generator()
        self.t = 0
        self.last_time = None

    def __iter__(self):
        return self

    def __next__(self):
        frame = next(self.frames)

        range_images, _, _, range_image_top_poses = frame_utils.parse_range_image_and_camera_projection(frame)

        # convert the range image to ranges
        ranges, points = self.__convert_range_image_to_2D(
            frame=frame,
            range_images=range_images,
            range_image_index=0,
            range_image_top_poses=range_image_top_poses,
            lidar_calibrations=frame.context.laser_calibrations,
        )

        # calculate the current yaw
        frame_transform = np.reshape(np.array(frame.pose.transform), [4, 4])

        # TODO: This is a hack to get the yaw from the transform matrix -- should be able to
        #       get this directly from the transform, but I don't know the rotation order.
        rotated = np.dot(frame_transform[0:3, 0:3], np.array([1, 0, 0]))
        yaw = np.arctan2(rotated[1], rotated[0]).astype(np.float32)

        dt = frame.timestamp_micros - self.last_time if self.last_time is not None else 0
        self.t += dt
        self.last_time = frame.timestamp_micros

        # get the positions (boxes) of the objects
        agents = []
        for obj in frame.laser_labels:
            agent_pos = np.array([obj.box.center_x, obj.box.center_y, obj.box.center_z, 1.0])
            agent_pos = np.dot(frame_transform, agent_pos)

            agent = {
                "id": obj.id,
                "centre": agent_pos[:3],
                "type": "VEHICLE" if obj.type == 1 else "PEDESTRIAN",
                "size": (obj.box.length, obj.box.width, obj.box.height),
                "yaw": obj.box.heading + yaw,
                "top_lidar_points": obj.num_top_lidar_points_in_box,
                "lidar_points": obj.num_lidar_points_in_box,
                "detection": obj.detection_difficulty_level,
                "tracking": obj.tracking_difficulty_level,
            }
            agents.append(agent)

        result = {
            "t": self.t,
            "dt": dt,
            "ranges": ranges,
            "points": points,
            "pos": (float(frame_transform[0, 3]), float(frame_transform[1, 3])),
            "yaw": yaw,
            "agents": agents,
        }
        return result

    def __convert_range_image_to_2D(
        self, frame, range_images, range_image_index, range_image_top_poses, lidar_calibrations
    ):
        """Convert the range images int bucketized lidar rays."""

        calibrations = sorted(lidar_calibrations, key=lambda c: c.name)

        num_buckets = int(np.pi * 2.0 / self.scan_params["INCREMENT"]) + 1

        # initialize a tensor to hold the ranges
        ranges = tf.fill([num_buckets], float(self.scan_params["RANGE"] + 1))

        cartesian_points = frame_utils.convert_range_image_to_cartesian(
            frame, range_images, range_image_top_poses, range_image_index, keep_polar_features=False
        )

        raw_points = None
        for calibration in calibrations:

            # TODO: Only using the top lidar -- had issues with the short range lidars showing
            #       nonexistant objects.
            if calibration.name != 1:
                continue

            # Filter out any points at or below the ground plane.
            # TODO: this is a hack to quickly eliminate the road surface from the point cloud.
            points = cartesian_points[calibration.name]

            # Create a mask for the height range
            points = tf.boolean_mask(points, (points[..., 2] > Z_MIN) & (points[..., 2] < Z_MAX))
            points = tf.reshape(points, [-1, 3])

            # rotate/translate to the world frame
            transform = tf.reshape(frame.pose.transform, [4, 4])
            rotation = transform[0:3, 0:3]
            translation = transform[0:3, 3]

            world_points = tf.einsum("ij,kj->ki", rotation, points) + translation

            if raw_points is None:
                raw_points = world_points
            else:
                raw_points = tf.concat([raw_points, world_points], axis=0)

            # Convert the points to lidar rays
            # use tf to find the ranges (ignore the z component of the points)
            point_ranges = tf.math.reduce_euclidean_norm(points[:, :2], axis=1)
            point_angles = (
                tf.clip_by_value((tf.atan2(points[:, 1], points[:, 0]) + np.pi), 0, np.pi * 2.0)
                / self.scan_params["INCREMENT"]
            )
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
        return ranges, raw_points

    def get_map(self, pos, size):
        # map = self.map.get_probability_at((pos[0] - (size * scale) / 2.0, pos[1] - (size * scale) / 2.0), (size, size))

        x_min = int(((pos[0] - size / 2.0) - self.map_origin[0]) * self.map_pixels_per_meter)
        x_max = int(x_min + size * self.map_pixels_per_meter)
        y_min = int(((pos[1] - size / 2.0) - self.map_origin[1]) * self.map_pixels_per_meter)
        y_max = int(x_min + size * self.map_pixels_per_meter)

        map = {
            "VEHICLE": np.max(self.scenario_map_data["VEHICLE"][:, x_min:x_max, y_min:y_max], axis=0),
            "PEDESTRIAN": np.max(self.scenario_map_data["PEDESTRIAN"][:, x_min:x_max, y_min:y_max], axis=0),
        }

        return map
