import json
import pickle
import tensorrt  # import before tensorflow to prevent TensorRT not found error
import tensorflow as tf
from typing import List

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
from scenario import Scenario


FIG_LIDAR_MAP = 1
FIG_OCCUPANCY_GRID = 2
FIG_DYNAMIC_OCCUPANCY_GRID = 3

DEFAULT_CACHE_LOC = "./cache"
CACHE_PATH = "v1/perception/1_4_3/training"
MAP_DIR = "maps"


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
    parser.add_argument(
        "--context", type=str, default=None, help="Context string.  If not provided, a random context is selected."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    return parser.parse_args()


def get_context(cache_location: str, rng: np.random.Generator):
    filenames = tf.io.gfile.glob(os.path.join(cache_location, CACHE_PATH, "*.tfrecord"))
    filename = rng.choice(filenames)
    return os.path.basename(filename)


def main():
    # parse arguments
    args = parse_args()

    # create a generator
    rng = np.random.default_rng(seed=args.seed)

    # load the dataset
    context = args.context if args.context is not None else get_context(args.cache_location, rng)
    scenario = Scenario(context, args.cache_location, {"INCREMENT": LIDAR_INCREMENT, "RANGE": LIDAR_RANGE})

    # set up some display parameters
    map_fig, map_ax = plt.subplots(1, 1, num=FIG_DYNAMIC_OCCUPANCY_GRID, figsize=(10, 10))
    map_im = map_ax.imshow(np.zeros([GRID_SIZE, GRID_SIZE, 3], dtype=np.uint8))

    ref_fig, ref_ax = plt.subplots(1, 1, num=FIG_OCCUPANCY_GRID, figsize=(10, 10))
    ref_im = ref_ax.imshow(np.zeros([GRID_SIZE, GRID_SIZE, 3], dtype=np.uint8))

    # bev_fig, bev_ax = plt.subplots(1, 1, num=FIG_OCCUPANCY_GRID, figsize=(10, 10))
    # bev_im = bev_ax.imshow(np.zeros([1000, 1000, 3], dtype=np.uint8))

    plt.show(block=False)

    # construct the occupancy grid
    (
        dogm,
        lmg,
    ) = construct_occupancy_grid()
    occupancy_grid = None

    for data in scenario:

        ranges = data["ranges"].numpy().astype(np.float32)
        points = data["points"].numpy().astype(np.float32)

        fig = plt.figure(FIG_LIDAR_MAP)
        plt.clf()
        plt.scatter(points[:, 0], points[:, 1], s=0.5)
        plt.xlim([-100, 100])
        plt.ylim([-100, 100])

        # update the occupancy grid
        grid_data = lmg.generateGrid(VectorFloat(ranges), np.rad2deg(data["yaw"]))

        measurements = renderMeasurement(grid_data).reshape(2, GRID_SIZE, GRID_SIZE)
        occupancy_grid = update_occupancy_grid(
            occupancy_grid=occupancy_grid,
            measurements=measurements,
            pose_data=(data["pos"][0], data["pos"][1], data["yaw"]),
        )

        dogm.updateGrid(grid_data, data["pos"][0], data["pos"][1], float(data["dt"] / 1.0e6))

        # render the occupancy grids
        occ_grid = occupancy_grid.probability_map()
        occ_img = Image.fromarray((occ_grid * 255).astype(np.uint8)).convert("RGB")
        ref_im.set_data(occ_img)

        dyn_occ_grid = renderOccupancyGrid(dogm)
        map_img = Image.fromarray(((1 - dyn_occ_grid) * 255).astype(np.uint8)).convert("RGB")
        map_im.set_data(map_img)

        plt.pause(0.2)

    map_data = {
        "context": context,
        "origin": list(occupancy_grid._origin),
        "height": occupancy_grid._height,
        "width": occupancy_grid._width,
        "resolution": occupancy_grid._resolution,
    }

    map_path = f"{args.cache_location}/{MAP_DIR}"
    # check if the map directory exists and create it if it does not
    if not os.path.exists(map_path):
        os.makedirs(map_path)

    # write the map data to a json file
    with open(f"{map_path}/{context}.json", "w") as map_file:
        json.dump(map_data, map_file)
    pickle.dump(occupancy_grid, open(f"{map_path}/{context}.pkl", "wb"))
    map_img.save(f"{map_path}/{context}.png")


if __name__ == "__main__":
    main()
