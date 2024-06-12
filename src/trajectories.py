# Experiments with loading data from the waymo open dataset
#
from typing import Optional
import warnings

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
np.set_printoptions(linewidth=120)

import os
import itertools
from math import ceil, sqrt, exp, log, acos, isnan, isinf
import random
import pandas as pd

from PIL import Image

from enum import Enum
from controller.controller import init_controller, get_control, get_bb_vertices
from controller.validate import (
    draw_agent,
    visualize_variations,
    visualize_controls,
    run_trajectory,
)

from controller.discrete_frechet import FastDiscreteFrechetMatrix, euclidean

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
from trajectory_planner.trajectory_planner import TrajectoryPLanner
from Grid.visibility_costmap import update_visibility_costmap
from Grid.visibility_costmap import VisibilityGrid

from config import *

MIN_COLLISION_DISTANCE = 2.0
TTC_THRESHOLD = 0.75
OCCLUSION_PENALTY = 5.0  # relatively large penalty for occlusion mismatch

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
    parser.add_argument(
        "--occlusion-rate",
        type=float,
        default=0.1,
        help="Probability of occlusion for each of the agent locations",
    )

    parser.add_argument("--output-dir", type=str, default=".", help="Location to drop output files")
    parser.add_argument(
        "--planning-time",
        metavar="horizon",
        default=2.5,
        type=float,
        help="Time horizon for planning (seconds)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of MPPI samples to generate for each control input",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.05,
        help="Time step discretization for simulation (seconds)",
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

    args = parser.parse_args()

    if args.test_mode == "trajectory":
        args.trials = 1

    return args

def check_collision( ego_trajectory, agent_trajectory ) -> Optional[int]:

    try:
        for i in range(len(ego_trajectory)):
            ego_state = ego_trajectory[i]
            agent_state = agent_trajectory[i]

            # BUGBUG - using a simple distance for now, but should replace with
            #          a more accurate collision detection algorithm
            distance = np.linalg.norm(ego_state[:2] - agent_state[:2])
            if distance < MIN_COLLISION_DISTANCE:
                return i
    except IndexError:
        pass

    return None


def calculate_simularity(t1, occ1, t2, occ2):

    # calculate the euclidean distance between the two trajectories
    euc_diff = 0.0
    for i in range(len(t1)):
        if occ1[i] and occ2[i]:
            # both occluded -- no information
            continue
        elif occ1[i] or occ2[i]:
            # one occluded -- should be different
            euc_diff += OCCLUSION_PENALTY
        else:
            euc_diff += np.linalg.norm(t1[i, 0:2] - t2[i, 0:2])
    euc_diff /= len(t1)

    # BUGBUG -- currently don't have a way to account for occlusions
    #           in frechet distance calculation
    #
    # # calculate the frechet distance between the two trajectories
    # frechet = FastDiscreteFrechetMatrix(euclidean)
    # frechet_distance = frechet.distance(np.array(t1[:, :2]), np.array(t2[:, :2]))

    return euc_diff, np.inf # frechet_distance


class Direction(Enum):
    LEFT = 0
    RIGHT = 1
    STRAIGHT = 2


def generate_controls(
    generator: np.random.Generator,
    direction: "tuple[Direction,float]",
    control_range: "tuple[float,float]",
    disturbance: float,
    steps: int,
) -> "list[list[float, float]]":
    controls = []
    direction, degree = direction
    if direction == Direction.LEFT:
        # left is a positive angle, turning counter-clockwise
        control_mean = control_range[1] * degree
    elif direction == Direction.RIGHT:
        control_mean = control_range[0] * degree
    else:
        control_mean = 0.0

    for _ in range(steps):
        noise = generator.normal(0.0, disturbance)
        controls.append([0.0, control_mean + noise])
    return controls


def generate_trajectories(
    controller,
    generator: np.random.Generator,
    initial_state: "list[float]",
    num_trajectories: int,
    control_range: "tuple[float,float]",
    directions: "list[tuple[Direction,float]]",
    probabilities: "list[float]",
    disturbance: float,
    steps: int,
    dt: float,
) -> "list[list[list[float, float]]]":
    trajectories = []
    control_sets = []

    for i in range(num_trajectories):
        direction = generator.choice(directions, p=probabilities)
        controls = generate_controls(
            generator=generator, direction=direction, control_range=control_range, disturbance=disturbance, steps=steps
        )
        control_sets.append(controls)

    for controls in control_sets:
        trajectory = run_trajectory(controller.vehicle, initial_state=initial_state, controls=controls, dt=dt)
        trajectories.append(trajectory)

    return trajectories, control_sets


def plot_trajectories(trajectories: "list[np.array]") -> None:
    fig, ax = plt.subplots()

    for trajectory in trajectories:
        plt.plot(trajectory[:, 0], trajectory[:, 1])

    ax.axis("equal")
    plt.savefig( "trajectories.png" )


def generate_occlusions( generator: np.random.Generator, occlusion_rate: float, trajectory: "list[list[float]]" ) -> "list[bool]":
    occlusions = []
    for i in range(len(trajectory)):
        # if trajectory[i][0] > 4.1:
        #     occluded = generator.choice([True, False], p=[occlusion_rate, 1.0 - occlusion_rate])
        # else:
        #     occluded = True
        occluded = generator.choice([True, False], p=[occlusion_rate, 1.0 - occlusion_rate])
        occlusions.append(occluded)
    return occlusions

def main():

    args = parse_args()

    rng = np.random.default_rng(seed=args.seed)

    if args.prefix is not None:
        args.prefix = f"{args.prefix}-"
    else:
        args.prefix = ""

    try:

        controller = init_controller(args)

        # generate a trajectory for the Ego vehicle
        ego_trajectories, ego_controls = generate_trajectories(
            controller=controller,
            generator=rng,
            initial_state=[0.0, 0.0, 5.0, 0.0],
            num_trajectories=1,
            control_range=(-np.pi / 5.0, np.pi / 5.0),
            directions=[(Direction.STRAIGHT, 1.0)],
            probabilities=[1.0],
            disturbance=0.00,
            steps=10,
            dt=args.time_step,
        )
        ego_trajectory = ego_trajectories[0]

        trajectories, controls = generate_trajectories(
            controller=controller,
            generator=rng,
            initial_state=[4.0, -4.0, 5.0, np.pi / 2.0],
            num_trajectories=10,
            control_range=(-np.pi / 5.0, np.pi / 5.0),
            directions=[
                (Direction.LEFT, 0.5),
                (Direction.LEFT, 0.3),
                (Direction.RIGHT, 0.9),
                (Direction.RIGHT, 0.7),
                (Direction.STRAIGHT, 1.0),
            ],
            probabilities=[0.3, 0.0, 0.3, 0.0, 0.4],
            # directions=[(Direction.STRAIGHT, 1.0)],
            # probabilities=[1.0],
            disturbance=0.1,
            steps=10,
            dt=args.time_step,
        )

        for occ in [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]:
            # generate occlusions for each trajectory
            occlusions = []
            for trajectory in trajectories:
                occlusions.append(generate_occlusions(rng, occlusion_rate=occ, trajectory=trajectory))

            plot_trajectories(trajectories=[ego_trajectory] + trajectories)

            # check for collisions
            collision_table = np.zeros([len(trajectories),])
            for i, trajectory in enumerate(trajectories):
                collision = check_collision(ego_trajectory, trajectory)
                collision_table[i] = collision
            print( collision_table )

            # Each trajectory now needs to be evaluated for visibility -- for now we assume
            # all are completely visible

            # calculate the probability of each trajectory based on the potential for observations
            # and the trajectory that occurs.

            trajectory_probabilities = []
            collision_probabilities = []
            for obs_trajectory, obs_occlusion in zip( trajectories, occlusions ):
                probability = np.zeros([len(trajectories), len(obs_trajectory) + 1])
                probability[:, 0] = np.ones(len(trajectories)) * (1.0 / len(trajectories))  # uniform distribution

                step_collision_probability = []
                for step in range(1, len(obs_trajectory)):
                    for target_index, (target_trajectory, target_occlusion) in enumerate(zip(trajectories,occlusions)):
                        euc_diff, _ = calculate_simularity(obs_trajectory[:step+1], obs_occlusion[:step+1],
                                                                      target_trajectory[:step+1], target_occlusion[:step+1])
                        probability[target_index, step] = np.exp(-euc_diff)
                    # normalize
                    probability[:, step] /= np.sum(probability[:, step])

                    collision_probability = 0
                    for target_index, target_trajectory in enumerate(trajectories):
                        ttc = (collision_table[target_index] - step) * args.time_step if collision_table[target_index] is not None else np.inf
                        if ttc < TTC_THRESHOLD:
                            collision_probability += probability[target_index, step]

                    step_collision_probability.append(collision_probability)

                trajectory_probabilities.append(probability)
                collision_probabilities.append(step_collision_probability)

            collision_probabilities = np.array(collision_probabilities)

            collision_probabilities = np.mean(collision_probabilities, axis=0)

            print(collision_probabilities)

    except KeyboardInterrupt:
        pass
    finally:

        print("Exiting...")


if __name__ == "__main__":
    main()