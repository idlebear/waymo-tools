# Experiments with loading data from the waymo open dataset
#
from typing import Optional
import warnings

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import os
import itertools
from math import ceil, sqrt
import random
import pandas as pd

from PIL import Image

import tensorrt
import tensorflow as tf

from tracker.tracker import Tracker

all_gpus = tf.config.experimental.list_physical_devices("GPU")
if all_gpus:
    try:
        for cur_gpu in all_gpus:
            # BUGBUG - hack to limit tensorflow memory usage
            # tf.config.experimental.set_memory_growth(cur_gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                cur_gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 12)],
            )

    except RuntimeError as e:
        print(e)

from controller.controller import init_controller, get_control, get_bb_vertices
from controller.validate import (
    draw_agent,
    visualize_variations,
    visualize_controls,
    run_trajectory,
)

from controller.discrete_frechet import FastDiscreteFrechetMatrix, euclidean

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Restrict TensorFlow to only allocate half of the first GPU's memory
        half_memory = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 12)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [half_memory])
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


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
from trajectory_planner.trajectory_planner import TrajectoryPlanner
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
    parser.add_argument("--output-dir", type=str, default=".", help="Location to drop output files")
    parser.add_argument(
        "--planning-time",
        metavar="horizon",
        default=2.5,
        type=float,
        help="Time horizon for planning (seconds)",
    )
    parser.add_argument("--trials", type=int, default=10, help="Repetitions for each context")
    parser.add_argument(
        "--c-lambda",
        type=float,
        default=DEFAULT_LAMBDA,
        help="Lambda value for weight normalization control",
    )
    parser.add_argument(
        "--method",
        metavar="method",
        default="None",
        action="store",
        choices=["Ignore", "None", "Ours", "Higgins", "Andersen"],
        help="Method options",
    )
    parser.add_argument(
        "--test-mode",
        default="trajectory",
        action="store",
        choices=[
            "mppi",
            "trajectory",
        ],
        help="Test types",
    )
    parser.add_argument(
        "--mppi-m",
        type=float,
        default=DEFAULT_METHOD_WEIGHT,
        help="M/Lambda value for method weights",
    )
    parser.add_argument("--x_weight", type=float, default=X_WEIGHT, help="Weight for x coordinate")
    parser.add_argument("--y_weight", type=float, default=Y_WEIGHT, help="Weight for y coordinate")
    parser.add_argument("--v_weight", type=float, default=V_WEIGHT, help="Weight for velocity")
    parser.add_argument("--theta_weight", type=float, default=THETA_WEIGHT, help="Weight for theta")
    parser.add_argument("--a_weight", type=float, default=A_WEIGHT, help="Weight for acceleration")
    parser.add_argument("--delta_weight", type=float, default=DELTA_WEIGHT, help="Weight for delta")
    parser.add_argument("--prefix", default=None, help="Prefix for output files")
    parser.add_argument("--discount", type=float, default=DISCOUNT_FACTOR, help="Weight for discount")

    # trajectron args
    parser.add_argument("--config", type=str, help="Configuration file for the model")
    parser.add_argument("--model-iteration", type=int, help="Model version")
    parser.add_argument("--model-dir", type=str, help="Location of the model files")
    parser.add_argument("--attention-radius", type=float, default=3.0, help="Model version")
    parser.add_argument("--device", type=str, default=None, help="Model version")
    parser.add_argument("--samples", type=int, default=5, help="Model version")
    parser.add_argument("--history-len", type=int, default=10, help="Model version")
    parser.add_argument("--horizon", type=int, default=10, help="Model prediction horizon")
    parser.add_argument("--incremental", help="Use Trajectron in online incremental mode", action="store_true")
    parser.add_argument("--results-dir", type=str, help="Location of generated output")

    # Model Parameters
    parser.add_argument(
        "--offline_scene_graph",
        help="whether to precompute the scene graphs offline, options are 'no' and 'yes'",
        type=str,
        default="yes",
    )

    parser.add_argument(
        "--dynamic_edges",
        help="whether to use dynamic edges or not, options are 'no' and 'yes'",
        type=str,
        default="yes",
    )

    parser.add_argument(
        "--edge_state_combine_method",
        help="the method to use for combining edges of the same type",
        type=str,
        default="sum",
    )

    parser.add_argument(
        "--edge_influence_combine_method",
        help="the method to use for combining edge influences",
        type=str,
        default="attention",
    )

    parser.add_argument(
        "--edge_addition_filter",
        nargs="+",
        help="what scaling to use for edges as they're created",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
    )  # We don't automatically pad left with 0.0, if you want a sharp
    # and short edge addition, then you need to have a 0.0 at the
    # beginning, e.g. [0.0, 1.0].

    parser.add_argument(
        "--edge_removal_filter",
        nargs="+",
        help="what scaling to use for edges as they're removed",
        type=float,
        default=[1.0, 0.0],
    )  # We don't automatically pad right with 0.0, if you want a sharp drop off like
    # the default, then you need to have a 0.0 at the end.

    parser.add_argument(
        "--override_attention_radius",
        action="append",
        help='Specify one attention radius to override. E.g. "PEDESTRIAN VEHICLE 10.0"',
        default=[],
    )

    parser.add_argument(
        "--incl_robot_node",
        help="whether to include a robot node in the graph or simply model all agents",
        action="store_true",
    )

    parser.add_argument("--map_encoding", help="Whether to use map encoding or not", action="store_true")

    parser.add_argument("--augment", help="Whether to augment the scene during training", action="store_true")

    parser.add_argument(
        "--node_freq_mult_train",
        help="Whether to use frequency multiplying of nodes during training",
        action="store_true",
    )

    parser.add_argument(
        "--node_freq_mult_eval",
        help="Whether to use frequency multiplying of nodes during evaluation",
        action="store_true",
    )

    parser.add_argument(
        "--scene_freq_mult_train",
        help="Whether to use frequency multiplying of nodes during training",
        action="store_true",
    )

    parser.add_argument(
        "--scene_freq_mult_eval",
        help="Whether to use frequency multiplying of nodes during evaluation",
        action="store_true",
    )

    parser.add_argument(
        "--scene_freq_mult_viz",
        help="Whether to use frequency multiplying of nodes during evaluation",
        action="store_true",
    )

    parser.add_argument("--no_edge_encoding", help="Whether to use neighbors edge encoding", action="store_true")


    args = parser.parse_args()

    if args.test_mode == "trajectory":
        args.trials = 1

    return args


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
        _high_x = min(grid_size - 1, _x + dx)
        _low_y = max(0, _y - dy)
        _high_y = min(grid_size - 1, _y + dy)

        xx, yy = np.meshgrid(range(_low_x, _high_x + 1), range(_low_y, _high_y + 1), indexing="xy")

        locations.update([(x, y) for x, y in zip(xx.flatten(), yy.flatten())])

    # print("\tGenerating locations to observe finished in", (time() - _s_time))
    return locations


def update_APCM(
    scenario,
    trajectory,
    visibility,
    occupancy,
    planning_time=5.0,
    planning_horizon=10,
    dt=0.1,
):
    """
    Update the APCM based on the current occupancy grid, visibility grid, and trajectory
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
    current_map = scenario.get_map((trajectory.x[0], trajectory.y[0]), grid_size)
    current_map = np.maximum(current_map["PEDESTRIAN"], current_map["VEHICLE"]).astype(np.float32) / 255.0
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
    # print(f"    Costmap update time: {costmap_update_time:.3f} seconds")

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


def calculate_simularity(controller, control1, control2, dt):

    t1 = run_trajectory(controller.vehicle, [0, 0, control1["v"], 0], control1["u"], dt)
    t2 = run_trajectory(controller.vehicle, [0, 0, control2["v"], 0], control2["u"], dt)

    # calculate the euclidean distance between the two trajectories
    euc_diff = 0.0
    for i in range(len(t1)):
        euc_diff += np.linalg.norm(t1[i][:2] - t2[i][:2])
    euc_diff /= len(t1)

    # calculate the frechet distance between the two trajectories
    frechet = FastDiscreteFrechetMatrix(euclidean)
    frechet_distance = frechet.distance(np.array(t1[:, :2]), np.array(t2[:, :2]))

    return euc_diff, frechet_distance


def compare_controls(controller, control_records, dt):

    traj_diffs = {
        "ours": [],
        "higgins": [],
        "andersen": [],
    }

    for control in zip(control_records["ours"][:-1], control_records["ours"][1:]):
        diffs = calculate_simularity(controller, control[0], control[1], dt)
        # print(f"Ours: Euclidean: {diffs[0]:.3f}, Frechet: {diffs[1]:.3f}")
        traj_diffs["ours"].append(diffs)
    traj_diffs["ours"] = np.array(traj_diffs["ours"])

    for control in zip(control_records["higgins"][:-1], control_records["higgins"][1:]):
        diffs = calculate_simularity(controller, control[0], control[1], dt)
        # print(f"Higgins: Euclidean: {diffs[0]:.3f}, Frechet: {diffs[1]:.3f}")
        traj_diffs["higgins"].append(diffs)
    traj_diffs["higgins"] = np.array(traj_diffs["higgins"])

    for control in zip(control_records["andersen"][:-1], control_records["andersen"][1:]):
        diffs = calculate_simularity(controller, control[0], control[1], dt)
        # print(f"Andersen: Euclidean: {diffs[0]:.3f}, Frechet: {diffs[1]:.3f}")
        traj_diffs["andersen"].append(diffs)
    traj_diffs["andersen"] = np.array(traj_diffs["andersen"])

    # t = [t + 1 for t in range(len(traj_diffs["ours"]))]

    # # create a graph of the euclidean differences and write it to a pdf
    # fig, ax = plt.subplots()
    # plt.plot(t, traj_diffs["ours"][:, 0], label="Ours")
    # plt.plot(t, traj_diffs["higgins"][:, 0], label="Higgins")
    # plt.plot(t, traj_diffs["andersen"][:, 0], label="Andersen")
    # plt.xlabel("Time")
    # plt.ylabel("Euclidean Difference")
    # plt.legend()
    # plt.savefig("euclidean_differences.pdf")
    # plt.close()

    # # create a plot of the frechet diffs as well
    # fig, ax = plt.subplots()
    # plt.plot(t, traj_diffs["ours"][:, 1], label="Ours")
    # plt.plot(t, traj_diffs["higgins"][:, 1], label="Higgins")
    # plt.plot(t, traj_diffs["andersen"][:, 1], label="Andersen")
    # plt.xlabel("Time")
    # plt.ylabel("Frechet Difference")
    # plt.legend()
    # plt.savefig("frechet_differences.pdf")
    # plt.close()

    # print the mean diff for each method
    mean_diff_ours = np.mean(traj_diffs["ours"], axis=0)
    mean_diff_higgins = np.mean(traj_diffs["higgins"], axis=0)
    mean_diff_andersen = np.mean(traj_diffs["andersen"], axis=0)
    print(f"Mean Difference (Ours) -- Euclidean: {mean_diff_ours[0]:.3f}, Frechet: {mean_diff_ours[1]}")
    print(f"Mean Difference (Higgins) -- Euclidean: {mean_diff_higgins[0]:.3f}, Frechet: {mean_diff_higgins[1]}")
    print(f"Mean Difference (Andersen) -- Euclidean: {mean_diff_andersen[0]:.3f}, Frechet: {mean_diff_andersen[1]}")

    return [mean_diff_ours, mean_diff_higgins, mean_diff_andersen]


def convert_visible_agents(pos, visible_agents):
    actors = []
    distances = []

    for agent in visible_agents:
        # for each agent, collect the center, bounding box, radius, closest point, and distance from the ego vehicle
        # BUGBUG -- this is a bit of a hack -- we're assuming the bounding box is a circle based on the width of the vehicle
        #           and not the worst case length.  We need to represent agents as a series circles, but for now this should
        #           be good enough.
        extent = 1.0  # min(agent["size"][0:2]) / 2.0
        x, y = agent["centre"][:2]
        distance = sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
        distances.append(distance)

        # Calculate the smallest angle to the bounding box -- used in the Andersen cost function
        vertices = get_bb_vertices(agent)
        min_pt = None
        min_angle = np.pi / 2.0
        for x, y in vertices:
            angle = abs(np.arctan((y - pos[1]) / (x - pos[0])))
            if angle < min_angle:
                min_angle = angle
                min_pt = (x, y)

        actors.append(
            [
                x,
                y,
                extent,
                min_pt[0],
                min_pt[1],
                sqrt((min_pt[0] - pos[0]) ** 2 + (min_pt[1] - pos[1]) ** 2),
            ]
        )
    # sort the actors by distance from the ego vehicle
    actors = [x for _, x in sorted(zip(distances, actors))]

    actors = np.array(actors)
    return actors


from math import exp, log, acos, isnan, isinf


def calculate_higgins_cost(agents, pos, scan_range):
    r_fov = scan_range
    r_fov_2 = r_fov * r_fov

    value = 0.0
    for agent in agents:
        dx = agent[0] - pos[0]
        dy = agent[1] - pos[1]
        d_2 = dx * dx + dy * dy
        d = sqrt(d_2)

        inner = agent[2] / d * (r_fov_2 - d_2)
        try:
            inner_exp = exp(inner)
            cost = log(1 + inner_exp)
        except OverflowError:
            cost = inner

        value += cost * cost

    # Higgins is a cost, but we sort in ascending order so the lowest value
    # will be ranked first
    return value


def calculate_andersen_cost(agents, pos, v):

    value = 0.0
    for agent in agents:
        dx = agent[3] - pos[0]
        dy = agent[4] - pos[1]

        dot = dx * v[0] + dy * v[1]
        if dot > 0:
            d = sqrt(dx * dx + dy * dy)
            v_mag = sqrt(v[0] * v[0] + v[1] * v[1])

            cost = acos(dot / (d * v_mag))
            value += cost
            break

    # Andersen is a reward -- we want to sort in descending order or just
    # invert the value
    return -value


def calculate_perception_gain(costmap, trajectory, agents, method):
    """
    Calculate the information gain of each trajectory based on the costmap

    """

    value = 0.0
    for i, (x, y) in enumerate(zip(trajectory.x, trajectory.y)):
        if i == 0:
            continue
        if method == "Ours":
            value += costmap.value(x, y)
            # we want the largest value to be first, so we negate the value
            value = -value
        elif method == "Higgins":
            value += calculate_higgins_cost(agents, (x, y), LIDAR_RANGE)
        elif method == "Andersen":
            v = (trajectory.x[i] - trajectory.x[i - 1], trajectory.y[i] - trajectory.y[i - 1])
            value += calculate_andersen_cost(agents, (x, y), v)
        else:
            raise ValueError(f"Unknown method: {method}")

    return value


def evaluate_trajectories(context, trial, step, costmap, trajectories, visible_agents, methods, fp):
    """
    Evaluate the trajectories based on the costmap
    """

    agents = convert_visible_agents((trajectories[0].x[0], trajectories[0].y[0]), visible_agents)

    for method in methods:
        values = []
        for initial_v, trajectory in enumerate(trajectories):
            value = calculate_perception_gain(costmap, trajectory, agents, method)
            values.append(value)

        # sort the values and get the index of the best trajectory
        order = np.argsort(values)
        ranking = np.zeros(len(trajectories), dtype=np.int32)
        for i, x in enumerate(order):
            ranking[x] = i

        for index, value, rank in zip(range(len(trajectories)), values, ranking):
            fp.write(f"{context}, {trial}, {step}, {method}, {index}, {value}, {value - values[0]}, {rank}\n")


def evaluate_controls(
    context,
    trial,
    step,
    controller,
    costmap,
    occupancy,
    u_nom,
    trajectory,
    target_speed,
    planning_horizon,
    dt,
    v,
    pos,
    yaw,
    visible_agents,
    methods,
    fp,
):
    controls = {}
    for method in methods:
        u, _ = get_control(
            controller=controller,
            costmap=costmap,
            occupancy=occupancy,
            u_nom=u_nom[method]["u"] if u_nom is not None else None,
            trajectory=trajectory,
            target_speed=target_speed,
            planning_horizon=planning_horizon,
            dt=dt,
            v=v,
            pos=pos,
            yaw=yaw,
            visible_agents=visible_agents,
            method=method,
        )
        controls[method] = {"u": u, "v": v}

        if u_nom is not None:
            euc, frech = calculate_simularity(controller, controls[method], u_nom[method], dt)
            fp.write(f"{context}, {trial}, {step}, {method}, {euc}, {frech}\n")

    return controls


def main():

    args = parse_args()

    # load the scenario
    context = args.context
    cache_location = args.cache_location
    scan_params = {"INCREMENT": np.pi * 2 / 2650.0, "RANGE": 75.0}

    rng = np.random.default_rng(seed=args.seed)

    if context is None:
        filenames = tf.io.gfile.glob(os.path.join(cache_location, CACHE_PATH, "*.tfrecord"))
        contexts = [os.path.basename(filename) for filename in filenames]
    else:
        contexts = [context]

    for context in contexts:
        print(f"Found scenario: {context}")

    if DEBUG_TRAJECTORIES:
        # set up some display parameters
        map_fig, map_ax = plt.subplots(1, 1, num=FIG_DYNAMIC_OCCUPANCY_GRID, figsize=(10, 10))
        map_im = map_ax.imshow(np.zeros([GRID_SIZE, GRID_SIZE, 3], dtype=np.uint8))

        bev_fig, bev_ax = plt.subplots(1, 1, num=FIG_LIDAR_MAP, figsize=(10, 10))
        bev_artist = None
        control_plot_data = {
            "figure": bev_fig,
            "ax": bev_ax,
            "plot_backgrounds": [],
        }
        plt.show(block=False)

    if args.prefix is not None:
        args.prefix = f"{args.prefix}-"
    else:
        args.prefix = ""

    if args.test_mode == "mppi":
        mppi_filename = f"{args.prefix}mppi_results.csv"
        mppi_title_str = "context,trial,step,method,euclidean,frechet\n"
        mppi_filename = os.path.join(args.output_dir, mppi_filename)
        mppi_fp = open(mppi_filename, "w")
        mppi_fp.write(mppi_title_str)

    traj_filename = f"{args.prefix}traj_results.csv"
    traj_title_str = "context,trial,step,method,trajectory,value,diff,order\n"
    traj_filename = os.path.join(args.output_dir, traj_filename)
    traj_fp = open(traj_filename, "w")
    traj_fp.write(traj_title_str)

    try:
        for context in contexts:
            print(f"Processing context: {context}")
            for trial in range(args.trials):
                try:
                    scenario = Scenario(context, cache_location, scan_params)
                except FileNotFoundError as e:
                    print(f"Failed to load scenario {context}")
                    continue

                trajectory = np.array(scenario.trajectory)

                planner = TrajectoryPlanner(trajectory)

                # construct the occupancy grid
                lmg = construct_laser_measurement_grid()
                dogm = construct_dynamic_occupancy_grid()
                occupancy_grid = None
                visibility_grid = VisibilityGrid(GRID_WIDTH, GRID_CELL_WIDTH, origin=(0, 0))

                # initialize a controller
                planning_horizon = int(args.planning_time * (1.0 / planner._dt))
                controller = init_controller(args)
                last_controls = None

                tracker = Tracker(
                    initial_timestep=0,
                    args=args,
                    scenario_map=scenario.scenario_map_data,
                    map_origin=scenario.map_origin,
                    map_pixels_per_meter=scenario.map_pixels_per_meter,
                    device="cuda:0",
                    dt=scenario.dt,
                )

                end = time.time()

                for step, data in enumerate(scenario):

                    start = time.time()
                    print(f"Step: {step} -- Enumeration Time: {start - end:.3f} seconds")

                    try:
                        working_trajectories = planner.generate_trajectories(
                            trajectories_requested=3,
                            planning_horizon=planning_horizon,
                            step=step,
                        )
                    except ValueError as e:
                        # finished trajectory
                        break

                    stage1 = time.time()
                    # print(f"    Trajectory generation time: {stage1 - start:.3f} seconds")

                    ranges = data["ranges"].numpy().astype(np.float32)
                    points = data["points"].numpy().astype(np.float32)

                    # update the occupancy grid
                    grid_data = lmg.generateGrid(VectorFloat(ranges), np.rad2deg(data["yaw"]))
                    dogm.updateGrid(
                        grid_data,
                        data["pos"][0],
                        data["pos"][1],
                        float(data["dt"] / 1.0e6),
                    )
                    dyn_occ_grid = renderOccupancyGrid(dogm)

                    # draw in the agents
                    tracker_update = []
                    for agent in data["agents"]:
                        tracker_update.append(
                            {
                                "id": agent["id"],
                                "type": agent["type"],
                                "pos": agent["centre"],
                                "size": agent["size"],
                                "heading": agent["yaw"],
                            }
                        )

                        draw_agent(
                            dyn_occ_grid,
                            data["pos"],
                            GRID_CELL_WIDTH,
                            agent["centre"],
                            agent["size"],
                            agent["yaw"],
                            min(1.0, agent["top_lidar_points"] / 1000),
                        )

                    tracker.step(tracker_update)

                    tracker.plot_predictions( frame=step, futures=None, results_dir="./results" )


                    stage2 = time.time()
                    # print(f"    Occupancy grid update time: {stage2 - stage1:.3f} seconds")

                    # update the APCM
                    update_APCM(
                        scenario=scenario,
                        trajectory=planner.get_planning_trajectory(),
                        visibility=visibility_grid,
                        occupancy=dyn_occ_grid,
                        planning_time=args.planning_time,
                        planning_horizon=planning_horizon,
                        dt=0.1,
                    )

                    stage3 = time.time()
                    # print(f"    Information gain calculation time: {stage3 - stage2:.3f} seconds")

                    if args.test_mode == "mppi":
                        controls = evaluate_controls(
                            context=context,
                            trial=trial,
                            step=step,
                            controller=controller,
                            costmap=visibility_grid,
                            occupancy=dyn_occ_grid,
                            u_nom=last_controls,
                            trajectory=planner.get_planning_trajectory(),
                            target_speed=working_trajectories[0].s_d[-1],
                            planning_horizon=planning_horizon,
                            dt=planner._dt,
                            v=working_trajectories[0].s_d[0],
                            pos=data["pos"],
                            yaw=data["yaw"],
                            visible_agents=data["agents"],
                            methods=["Ours", "Higgins", "Andersen"],
                            fp=mppi_fp,
                        )

                        if DEBUG_TRAJECTORIES:

                            nom_traj = np.array(
                                [
                                    [x, y]
                                    for x, y in zip(
                                        working_trajectories[0].x,
                                        working_trajectories[0].y,
                                    )
                                ]
                            )

                            if bev_artist is None:
                                bev_artist = bev_ax.scatter(points[:, 0], points[:, 1], s=4, animated=False)
                                bev_ax.draw_artist(bev_ax)

                            else:
                                bev_artist.set_offsets(points[:, :2])
                                bev_ax.draw_artist(bev_artist)

                            control_plot_data = visualize_controls(
                                plot_data=control_plot_data,
                                vehicle=controller.vehicle,
                                initial_state=(
                                    *data["pos"],
                                    working_trajectories[0].s_d[0],
                                    data["yaw"],
                                ),
                                nom_traj=nom_traj,
                                controls=controls,
                                dt=planner._dt,
                            )

                            bev_fig.canvas.blit(bev_ax.bbox)

                            bev_ax.set_xlim(
                                data["pos"][0] - LIDAR_RANGE / 4,
                                data["pos"][0] + LIDAR_RANGE / 4,
                            )
                            bev_ax.set_ylim(
                                data["pos"][1] - LIDAR_RANGE / 4,
                                data["pos"][1] + LIDAR_RANGE / 4,
                            )
                            bev_fig.canvas.draw()
                            bev_fig.canvas.flush_events()

                            # map = scenario.get_map(
                            #     (data["pos"][0], data["pos"][1]), GRID_SIZE, GRID_CELL_WIDTH
                            # )
                            # occ_img = Image.fromarray((map * 255).astype(np.uint8)).convert("RGB")
                            # ref_im.set_data(occ_img)

                            map_img = Image.fromarray(((1 - np.flipud(dyn_occ_grid)) * 255).astype(np.uint8)).convert(
                                "RGB"
                            )
                            map_im.set_data(map_img)

                        last_controls = controls

                    # evaluate the working trajectories based on the costmap
                    #
                    # only evaluate the first one as they don't change / there is no randomness
                    if trial == 0:
                        evaluate_trajectories(
                            context=context,
                            trial=trial,
                            step=step,
                            costmap=visibility_grid,
                            trajectories=working_trajectories,
                            visible_agents=data["agents"],
                            methods=["Ours", "Higgins", "Andersen"],
                            fp=traj_fp,
                        )

                    stage4 = time.time()
                    # print(f"    Control search time: {stage4 - stage3:.3f} seconds")

                    now = time.time()
                    # print(f"    Drawing time: {now - stage4:.3f} seconds")
                    print(f"Total time: {(now - end):.3f} seconds")
                    end = now

                    plt.pause(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # if args.test_mode == "mppi":
        #     mppi_fp.close()
        traj_fp.close()


if __name__ == "__main__":
    main()
