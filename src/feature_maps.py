from enum import Enum
from typing import List


import numpy as np
import os
import itertools
from math import ceil, sqrt
import random
import pandas as pd
import cv2


from PIL import Image

import tensorrt
import tensorflow as tf

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils.plot_maps import FeatureType

MAP_MARGIN = 0
LANE_WIDTH = 4.2
WALKWAY_WIDTH = 3.0


def plot_driving_lanes(
    map_features: List[map_pb2.MapFeature],
    layers: List[FeatureType] = None,
    pixels_per_meter: int = 1,
) -> np.array:
    """Plots the lanes from Scenario proto from the open dataset.

    Args:
    map_features: A list of map features to be plotted.

    Returns:
    A layered numpy array with each layer corresponding to a desired feature type.
    """

    if len(map_features) == 0:
        raise ValueError("No map features found in the scenario.")

    def add_points(points: List[map_pb2.MapPoint], is_polygon=False) -> np.array:
        feature_points = []
        for point in points:
            feature_points.append([point.x, point.y, point.z])
        if is_polygon:
            feature_points.append(feature_points[0])
        return np.array(feature_points)

    map_data = []

    for feature in map_features:
        if feature.HasField("lane"):
            data = add_points(list(feature.lane.polyline))
            map_data.append((data, FeatureType.SURFACE_STREET_LANE))

        elif feature.HasField("crosswalk"):
            data = add_points(
                list(feature.crosswalk.polygon),
                True,
            )
            map_data.append((data, FeatureType.CROSSWALK))
        elif feature.HasField("driveway"):
            data = add_points(
                list(feature.driveway.polygon),
                True,
            )
            map_data.append((data, FeatureType.DRIVEWAY))

    min_x = np.min([np.min(data[:, 0]) for data, _ in map_data]) - MAP_MARGIN
    max_x = np.max([np.max(data[:, 0]) for data, _ in map_data]) + MAP_MARGIN
    min_y = np.min([np.min(data[:, 1]) for data, _ in map_data]) - MAP_MARGIN
    max_y = np.max([np.max(data[:, 1]) for data, _ in map_data]) + MAP_MARGIN

    height = ceil(pixels_per_meter * (max_y - min_y))
    width = ceil(pixels_per_meter * (max_x - min_x))
    origin = np.array([min_x, min_y])

    # Create a blank image.
    if layers is None:
        layers = [
            [feature_type for _, feature_type in map_data],
        ]

    map_layers = np.zeros((len(layers), height, width), dtype=np.uint8)

    for layer_num, layer_group in enumerate(layers):
        for data, feature_type in map_data:
            if feature_type in layer_group:
                if feature_type == FeatureType.SURFACE_STREET_LANE:
                    # Lanes are represented as centerline polylines.  To draw the lane, we draw a rectangle
                    # around each point, rotated by the angle from the previous step.
                    for i in range(data.shape[0] - 1):
                        if i == 0:
                            theta = np.arctan2(data[i + 1, 1] - data[i, 1], data[i + 1, 0] - data[i, 0])
                        else:
                            theta = np.arctan2(data[i, 1] - data[i - 1, 1], data[i, 0] - data[i - 1, 0])
                        box = np.array(
                            [
                                [-LANE_WIDTH / 2, -LANE_WIDTH / 2],
                                [LANE_WIDTH / 2, -LANE_WIDTH / 2],
                                [LANE_WIDTH / 2, LANE_WIDTH / 2],
                                [-LANE_WIDTH / 2, LANE_WIDTH / 2],
                            ]
                        )
                        # rotate the box
                        box = (
                            np.dot(box, np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]))
                            + data[i, :2]
                        )
                        box = (pixels_per_meter * (box - origin)).astype(int)
                        # draw the box
                        cv2.fillPoly(map_layers[layer_num, ...], [box], 1)

                elif feature_type == FeatureType.CROSSWALK or feature_type == FeatureType.DRIVEWAY:
                    poly = (pixels_per_meter * (data[:, :2] - origin)).astype(int)
                    cv2.fillPoly(map_layers[layer_num, ...], [poly], 1)

    return map_layers, origin


def create_maps(map_features: List[map_pb2.MapFeature], pixels_per_meter: int = 1):

    map_layers, origin = plot_driving_lanes(
        map_features,
        layers=[
            [
                FeatureType.SURFACE_STREET_LANE,
                FeatureType.DRIVEWAY,
            ],
            [
                FeatureType.CROSSWALK,
                FeatureType.DRIVEWAY,
            ],
        ],
        pixels_per_meter=pixels_per_meter,
    )

    pedestrian_layer = cv2.dilate(map_layers[0, ...], np.ones((3, 3)), iterations=int(pixels_per_meter * WALKWAY_WIDTH))
    pedestrian_layer = pedestrian_layer - map_layers[0, ...]
    map_layers[1, ...] = pedestrian_layer | map_layers[1, ...]

    # rescale the map layers to 0-255
    map_layers = (255 * map_layers).astype(np.uint8)

    return map_layers, origin
