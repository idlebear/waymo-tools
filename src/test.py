# Experiments with loading data from the waymo open dataset
#
from typing import Optional
import warnings

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import os
import itertools
from math import ceil
import random

from PIL import Image

import tensorrt
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Restrict TensorFlow to only allocate half of the first GPU's memory
        half_memory = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 12)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [half_memory])
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# # # Create a configuration object with the desired settings
# # config = tf.compat.v1.ConfigProto()

# # # # Get the default TensorFlow session
# # # sess = tf.compat.v1.Session()

# # # # Print the number of threads used for parallelism
# # # print("Intra op parallelism threads: ", sess._config.intra_op_parallelism_threads)
# # # print("Inter op parallelism threads: ", sess._config.inter_op_parallelism_threads)

# # config.intra_op_parallelism_threads = 2  # Number of threads to use for intra-op parallelism
# # config.inter_op_parallelism_threads = 2  # Number of threads to use for inter-op parallelism

# # Create a session with the specified configuration
# sess = tf.compat.v1.Session(config=config)

import dask.dataframe as dd
from waymo_open_dataset import v2

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.v2.perception.utils import lidar_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d
import time
import argparse

from dogm_py import LaserMeasurementGridParams
from dogm_py import LaserMeasurementGrid
from dogm_py import DOGMParams
from dogm_py import DOGM
from dogm_py import VectorFloat
from dogm_py import renderOccupancyGrid
from dogm_py import renderMeasurement

from Grid.GridMap import ProbabilityGrid
from scenario import Scenario
from trajectory_planner.trajectory_planner import TrajectoryPLanner
from Grid.visibility_costmap import update_visibility_costmap
from Grid.visibility_costmap import VisibilityGrid

from config import *


def parse_args():
    parser = argparse.ArgumentParser(description="Experiments with loading data from the waymo open dataset")
    parser.add_argument(
        "--cache-location",
        type=str,
        default=DEFAULT_CACHE_LOC,
        help="Path to the directory with all components",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generation")
    parser.add_argument("--context", type=str, default=None, help="Name of the context")

    return parser.parse_args()


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


def construct_dynamic_occupancy_grid():
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

    return dogm


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


def update_pedestrian_locations_to_observe(trajectory, grid_size, planning_time=5.0):
    """
    _update_locations_to_observe:  use the current occupancy estimate, map and intended nominal trajectory
    to estimate which regions of the upcoming road are more important for viewing
    """

    _s_time = time.time()

    locations = set()

    for index in range(len(trajectory.x)):
        t = trajectory.t[index] - trajectory.t[0]
        # if t > planning_time:
        #     break

        dx = max(1, ceil(t * MAX_PEDESTRIAN_SPEED / GRID_CELL_WIDTH))
        dy = dx

        _x = int((trajectory.x[index] - trajectory.x[0]) / GRID_CELL_WIDTH + grid_size // 2)
        _y = int((trajectory.y[index] - trajectory.y[0]) / GRID_CELL_WIDTH + grid_size // 2)

        _low_x = max(0, _x - dx)
        _high_x = min(grid_size, _x + dx)
        _low_y = max(0, _y - dy)
        _high_y = min(grid_size, _y + dy)

        xx, yy = np.meshgrid(range(_low_x, _high_x + 1), range(_low_y, _high_y + 1), indexing="xy")

        locations.update([(x, y) for x, y in zip(xx.flatten(), yy.flatten())])

    # print("\tGenerating locations to observe finished in", (time() - _s_time))
    return locations


def calculate_information_gain(
    scenario, trajectory, visibility, occupancy, planning_time=5.0, planning_horizon=10, dt=0.1
):
    """
    Calculate the information gain of each trajectory
    """

    grid_size = GRID_SIZE * 2

    # calculate the locations to observe for each time step in frenet frame
    ped_locations = update_pedestrian_locations_to_observe(
        trajectory=trajectory, grid_size=grid_size, planning_time=dt * planning_horizon
    )
    car_locations = set()  # self._update_vehicle_locations_to_observe(grid_size=grid_size)

    world_locations_to_observe = list(car_locations.union(ped_locations))

    # update the map based on the current position, lidar map, and occupancy grid
    filtered_occupancy = np.where(occupancy < OCCUPANCY_THRESHOLD, 0, occupancy)

    # draw in the agents
    # for agent in self._visible_agents:
    #     draw_agent(filtered_occupancy, self._pos, GRID_CELL_WIDTH, agent)

    # the current map removes all the unseen areas of the occupancy grid
    current_map = scenario.get_map((trajectory.x[0], trajectory.y[0]), grid_size, GRID_CELL_WIDTH)
    current_map[
        int(grid_size / 2 - GRID_SIZE / 2) : int(grid_size / 2 + GRID_SIZE / 2),
        int(grid_size / 2 - GRID_SIZE / 2) : int(grid_size / 2 + GRID_SIZE / 2),
    ] = np.maximum(
        filtered_occupancy,
        current_map[
            int(grid_size / 2 - GRID_SIZE / 2) : int(grid_size / 2 + GRID_SIZE / 2),
            int(grid_size / 2 - GRID_SIZE / 2) : int(grid_size / 2 + GRID_SIZE / 2),
        ],
    )

    # build an extended map for the current position including uncertainty
    target_map = current_map.copy()
    target_map[
        int(grid_size / 2 - GRID_SIZE / 2) : int(grid_size / 2 + GRID_SIZE / 2),
        int(grid_size / 2 - GRID_SIZE / 2) : int(grid_size / 2 + GRID_SIZE / 2),
    ] = np.maximum(
        occupancy,
        target_map[
            int(grid_size / 2 - GRID_SIZE / 2) : int(grid_size / 2 + GRID_SIZE / 2),
            int(grid_size / 2 - GRID_SIZE / 2) : int(grid_size / 2 + GRID_SIZE / 2),
        ],
    )

    ends = [pt for pt in world_locations_to_observe if target_map[pt[1], pt[0]] > RISK_THRESHOLD]

    tic = time.time()
    obs_pts = update_visibility_costmap(
        costmap=visibility,
        map=current_map,
        centre=(trajectory.x[0], trajectory.y[0]),
        obs_trajectory=trajectory,
        lane_width=LANE_WIDTH,
        target_pts=ends,  # self._world_locations_to_observe,
    )
    costmap_update_time = time.time() - tic
    print(f"    Costmap update time: {costmap_update_time:.3f} seconds")

    if DEBUG_INFORMATION_GAIN:

        map_img = np.tile(np.expand_dims((current_map * 255.0).astype(np.uint8), axis=2), 3)
        # paint the target points red
        for x, y in world_locations_to_observe:
            if x > 0 and y > 0 and x < grid_size and y < grid_size:
                map_img[y, x, 0] = 255

        # paint the observation points green
        for x, y in obs_pts:
            x += int(GRID_SIZE / 2)
            y += int(GRID_SIZE / 2)
            if x > 0 and y > 0 and x < grid_size and y < grid_size:
                map_img[y, x, 1] = 255

        for x, y in zip(trajectory.x, trajectory.y):
            x = int((x - trajectory.x[0]) / GRID_CELL_WIDTH + grid_size // 2)
            y = int((y - trajectory.y[1]) / GRID_CELL_WIDTH + grid_size // 2)
            if x > 0 and y > 0 and x < grid_size and y < grid_size:
                map_img[y, x, :] = 255

        fig = plt.figure(FIG_VISIBILITY)
        plt.imshow(np.flipud(map_img))
        plt.show(block=False)


def draw_agent(map, origin, resolution, centre, size, yaw, visibility):

    size_y, size_x = map.shape

    x = int((centre[0] - origin[0]) / resolution + size_x // 2)
    y = int((centre[1] - origin[1]) / resolution + size_y // 2)

    # calculate the size of the rectangle in grid cells
    half_x = int(np.ceil(size[0] / (2.0 * resolution)))
    half_y = int(np.ceil(size[1] / (2.0 * resolution)))

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    for dx in range(-half_x, half_x + 1):
        for dy in range(-half_y, half_y + 1):
            _x = int(x + dx * cos_yaw - dy * sin_yaw)
            _y = int(y + dx * sin_yaw + dy * cos_yaw)
            if _x >= 0 and _x < GRID_SIZE and _y >= 0 and _y < GRID_SIZE:
                map[_y, _x] = visibility


def main():

    args = parse_args()

    # load the scenario
    context = args.context
    cache_location = args.cache_location
    scan_params = {"INCREMENT": np.pi * 2 / 2650.0, "RANGE": 75.0}

    rng = np.random.default_rng(seed=args.seed)

    if context is None:
        print("Selecting a random context.")
        filenames = tf.io.gfile.glob(os.path.join(cache_location, CACHE_PATH, "*.tfrecord"))
        filename = rng.choice(filenames)
        # update the context by taking the base name and extension from the path
        context = os.path.basename(filename)

    scenario = Scenario(context, cache_location, scan_params)

    trajectory = np.array(scenario.trajectory)

    planner = TrajectoryPLanner(trajectory)

    if DEBUG_TRAJECTORIES:
        # set up some display parameters
        map_fig, map_ax = plt.subplots(1, 1, num=FIG_DYNAMIC_OCCUPANCY_GRID, figsize=(10, 10))
        map_im = map_ax.imshow(np.zeros([GRID_SIZE, GRID_SIZE, 3], dtype=np.uint8))

        # ref_fig, ref_ax = plt.subplots(1, 1, num=FIG_OCCUPANCY_GRID, figsize=(10, 10))
        # ref_im = ref_ax.imshow(np.zeros([GRID_SIZE, GRID_SIZE, 3], dtype=np.uint8))

        bev_fig, bev_ax = plt.subplots(1, 1, num=FIG_LIDAR_MAP, figsize=(10, 10))
        bev_artist = None

        # # # plot the ground truth trajectory and orientations of the vehicle
        # # traj_fig, traj_ax = plt.subplots(1, 1, num=FIG_TRAJECTORIES, figsize=(10, 10))

        # traj_ax.plot(trajectory[:, 2], trajectory[:, 3], "r-")
        # for i in range(0, len(trajectory), 10):
        #     x = trajectory[i, 2]
        #     y = trajectory[i, 3]
        #     theta = trajectory[i, 4]
        #     traj_ax.arrow(x, y, 2 * np.cos(theta), 2 * np.sin(theta), head_width=0.1)

        # traj_ax.set_aspect("equal")
        # plt.show(block=False)
        # plt.pause(0.1)

    # construct the occupancy grid
    lmg = construct_laser_measurement_grid()
    dogm = construct_dynamic_occupancy_grid()
    occupancy_grid = None
    visibility_grid = VisibilityGrid(GRID_WIDTH, GRID_CELL_WIDTH, origin=(0, 0))

    stage4 = time.time()

    for step, data in enumerate(scenario):

        start = time.time()
        print(f"Step: {step} -- Enumeration Time: {start - stage4:.3f} seconds")

        try:
            working_trajectories = planner.generate_trajectories(
                trajectories_requested=3, planning_horizon=int(3 * (1.0 / planner._dt)), step=step
            )
        except ValueError as e:
            # finished trajectory
            break

        stage1 = time.time()
        print(f"    Trajectory generation time: {stage1 - start:.3f} seconds")

        ranges = data["ranges"].numpy().astype(np.float32)
        points = data["points"].numpy().astype(np.float32)

        # update the occupancy grid
        grid_data = lmg.generateGrid(VectorFloat(ranges), np.rad2deg(data["yaw"]))
        dogm.updateGrid(grid_data, data["pos"][0], data["pos"][1], float(data["dt"] / 1.0e6))
        dyn_occ_grid = renderOccupancyGrid(dogm)

        # draw in the agents
        for agent in data["agents"]:
            draw_agent(
                dyn_occ_grid,
                data["pos"],
                GRID_CELL_WIDTH,
                agent["centre"],
                agent["size"],
                agent["yaw"],
                min(1.0, agent["top_lidar_points"] / 1000),
            )

        stage2 = time.time()
        print(f"    Occupancy grid update time: {stage2 - stage1:.3f} seconds")

        # update the APCM
        calculate_information_gain(
            scenario=scenario,
            trajectory=planner.get_planning_trajectory(),
            visibility=visibility_grid,
            occupancy=dyn_occ_grid,
            planning_time=3.0,
            planning_horizon=30,
            dt=0.1,
        )

        stage3 = time.time()
        print(f"    Information gain calculation time: {stage3 - stage2:.3f} seconds")

        if DEBUG_TRAJECTORIES:
            # get the current map section
            map = scenario.get_map((data["pos"][0], data["pos"][1]), GRID_SIZE, GRID_CELL_WIDTH)

            # ---------------------------------------------------------
            # Draw updates
            # ---------------------------------------------------------
            if bev_artist is None:
                bev_artist = bev_ax.scatter(points[:, 0], points[:, 1], s=2, animated=True)
                # plot the working trajectories
                bev_traj_artist = []
                for traj in working_trajectories:
                    artist = bev_ax.plot(traj.x, traj.y, "b-", animated=True)
                    bev_traj_artist.append(artist)

                plt.pause(0.1)
                # bg = bev_fig.canvas.copy_from_bbox(bev_ax.bbox)
                bev_ax.draw_artist(bev_ax)
                bev_fig.canvas.blit(bev_ax.bbox)

            else:
                bev_artist.set_offsets(points[:, :2])

                for i, traj in enumerate(working_trajectories):
                    bev_traj_artist[i][0].set_data(traj.x, traj.y)
                # bev_fig.canvas.restore_region(bg)
                bev_ax.draw_artist(bev_artist)
                for artist in bev_traj_artist:
                    bev_ax.draw_artist(artist[0])
                bev_fig.canvas.blit(bev_ax.bbox)
                # bev_fig.canvas.flush_events()

            # render the occupancy grids
            # occ_img = Image.fromarray((map * 255).astype(np.uint8)).convert("RGB")
            # ref_im.set_data(occ_img)

            map_img = Image.fromarray(((1 - np.flipud(dyn_occ_grid)) * 255).astype(np.uint8)).convert("RGB")
            map_im.set_data(map_img)

            now = time.time()
            print(f"    Drawing time: {now - stage3:.3f} seconds")
            print(f"Total time: {(now - stage4):.3f} seconds")
            stage4 = now

        plt.pause(0.1)


if __name__ == "__main__":
    main()
