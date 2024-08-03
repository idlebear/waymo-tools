# Using the APCM, evaluate the probable accupancy/uncertainty of the trajectory

import numpy as np
from math import ceil, sqrt
import pandas as pd
import time
import matplotlib.pyplot as plt

from config import GRID_CELL_WIDTH, GRID_SIZE, GRID_WIDTH, DEBUG_INFORMATION_GAIN, FIG_VISIBILITY

from controller.validate import draw_agent_in_occupancy
from Grid.visibility_costmap import get_visibility_dictionary, update_visibility_costmap


def get_agent_footprint( agent, trajectory_index=0, origin=(0,0), resolution=0.1, prediction_num=1 ):
    """
    Given an agent, return the locations that the agent can observe at the current time step for the given trajectory
    """
    predictions = agent["predictions"]
    if len(predictions) == 0:
        return []

    # get the current position of the agent
    center = predictions[trajectory_index, prediction_num, :2]
    yaw = predictions[trajectory_index, prediction_num, 2]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    x = int((center[0] - origin[0]) / resolution)
    y = int((center[1] - origin[1]) / resolution)

    # calculate the size of the rectangle in grid cells
    size = agent["size"]
    half_x = int(np.ceil(size[0] / (2.0 * resolution)))
    half_y = int(np.ceil(size[1] / (2.0 * resolution)))

    rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

    # Create a grid of points around the agent
    dx = np.arange(-half_x, half_x + 1)
    dy = np.arange(-half_y, half_y + 1)
    grid_x, grid_y = np.meshgrid(dx, dy)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Apply the rotation matrix to each point
    rotated_points = np.dot(grid_points, rotation_matrix.T)

    # Translate the rotated points to the correct position
    translated_points = rotated_points + np.array([x, y])

    return translated_points.astype(int).tolist()



def get_perception_dictionary(
    scenario,
    trajectory,
    visibility,
    agents,
    target_agent,
    prediction_num,
    beliefs
):
    """
    Update the APCM based on the current occupancy grid, visibility grid, and trajectory

    Arguments:
    scenario: the scenario object
    trajectory: the current trajectory
    visibility: the visibility grid
    map: the current map
    agents: the list of agent predictions
    prediction_num: the index number of the prediction
    """

    grid_size = GRID_SIZE * 2

    # TODO: focus on using the stored map first and ignore the lidar data.  There are two choices here: we
    #       either assume that there is no street furniture/occlusions in the scene and just start with a blank map
    #       or we start with the current map drawn from LIDAR data and update it with the current occupancy grid

    current_map = scenario.get_map(trajectory[0], grid_size)
    resolution = scenario.map._resolution

    map_x_size, map_y_size = current_map.shape
    map_origin =  (trajectory[0,0] - (map_x_size * resolution) / 2.0,
                   trajectory[0,1] - (map_y_size * resolution) / 2.0)

    agent_target = get_agent_footprint(target_agent, origin=map_origin, resolution=resolution, time_step=prediction_num)

    # draw in the agents
    for agent in agents:
        if agent == target_agent:
            continue

        # TODO: Assume that we can not see thru agents, though it appears that in some cases we can
        N, K, D = agent["predictions"].shape
        for n in range(N):
            draw_agent_in_occupancy(current_map, origin=map_origin, resolution=scenario.map._resolution,
                                    centre=agent["predictions"][n, prediction_num, :2], size=agent["size"],
                                    yaw=agent["predictions"][n, prediction_num, 2], visibility=beliefs[agent["id"][n]])

    # TODO: we could reduce the number of observations made by limiting the observation points to those on the trajectory
    #       in the interval in question.  For now, get them all and sort it out later.
    obs_pts = [ [int((pt[0] - map_origin[0]) / scenario.map._resolution),
                 int((pt[1] - map_origin[1]) / scenario.map._resolution)] for pt in trajectory ]

    tic = time.time()
    visibility_dictionary = get_visibility_dictionary(current_map, origin=map_origin, resolution=scenario.map._resolution, obs_trajectory=obs_pts, target_pts=agent_target)
    costmap_update_time = time.time() - tic
    print(f"    Perception update time: {costmap_update_time:.3f} seconds")

    return visibility_dictionary

def evaluate_trajectory( scenario, trajectory, agents, num_predictions=1, prediction_iterval=0.1, dt=0.1 ):
    """
    Evaluate the trajectory using the APCM

    Arguments:
    scenario: the scenario object
    trajectory: the current trajectory
    agents: the list of agent predictions
    """

    beliefs = {}
    visibility = {}


    for agent in agents:

        prediction = agent.get_predictions()
        N, K, D = prediction.shape

        # initialize trajectory beliefs for each agent
        beliefs[agent["id"]] = np.ones(num_predictions)/num_predictions

        # for each prediction request
        for k in range(K):

            tic = time.time()
            visibility_dictionary = get_perception_dictionary(scenario=scenario, trajectory=trajectory,
                                                              visibility=None, agents=agents, target=agent, prediction_num=k, beliefs=beliefs,
                                                              prediction_interval=prediction_iterval, dt=dt )
            visibility[agent["id"]] = visibility_dictionary
            perception_time = time.time() - tic
            print(f"Perception time: {perception_time:.3f} seconds")

            for obs_idx in range()


    # evaluate the trajectory
    information_gain = 0.0
    for key, value in visibility.items():
        information_gain += value

    return information_gain
