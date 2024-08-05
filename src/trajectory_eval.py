# Using the APCM, evaluate the probable accupancy/uncertainty of the trajectory

import numpy as np
from math import ceil, sqrt
import pandas as pd
import time
import matplotlib.pyplot as plt

from config import GRID_CELL_WIDTH, GRID_SIZE, GRID_WIDTH, DEBUG_INFORMATION_GAIN, FIG_VISIBILITY

from controller.validate import draw_agent_in_occupancy
from Grid.visibility_costmap import get_visibility_dictionary, update_visibility_costmap

from tracker.agent_track import AgentTrack

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
    target_agent_trajectory,
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

    # get the target agent's footprint -- there are going to be duplicates, so use a set and then convert to a list
    agent_target = get_agent_footprint(target_agent, trajectory_index=target_agent_trajectory, origin=map_origin, resolution=resolution,
                                                          prediction_num=prediction_num)

    # draw in the agents
    for agent in agents:
        if agent == target_agent:
            continue

        N, K, D = agent["predictions"].shape
        for n in range(N):
            draw_agent_in_occupancy(current_map, origin=map_origin, resolution=scenario.map._resolution,
                                    centre=agent["predictions"][n, prediction_num, :2], size=agent["size"],
                                    yaw=agent["predictions"][n, prediction_num, 2], visibility=beliefs[agent["id"][n]])

    # TODO: we could reduce the number of observations made by limiting the observation points to those on the trajectory
    #       in the interval in question.  For now, get them all and sort it out later.
    #
    # TODO: make sure that the time-step is used instead of the prediction number
    obs_pts = [ [int((pt[0] - map_origin[0]) / scenario.map._resolution),
                 int((pt[1] - map_origin[1]) / scenario.map._resolution)] for pt in trajectory ]

    tic = time.time()
    visibility_dictionary = get_visibility_dictionary(current_map, origin=map_origin, resolution=scenario.map._resolution, obs_trajectory=obs_pts, target_pts=agent_target)
    costmap_update_time = time.time() - tic
    print(f"    Perception update time: {costmap_update_time:.3f} seconds")

    return visibility_dictionary

def calculate_similarity( agent, trajectories, prediction_idx ):
    """
    Calculate the similarity of the current prediction to the previous prediction
    """

    similarity = []
    for t1 in trajectories:
        obs = t1[prediction_idx, :2]

        trajectory_similarity = []
        for t2 in trajectories:
            pose = t2[prediction_idx, :2]
            diff = np.linalg.norm(obs - pose)
            trajectory_similarity.append(np.exp(-diff))

        similarity.append(trajectory_similarity)

    # normalize the similarity
    similarity = np.array(similarity)
    similarity = similarity / np.sum(similarity, axis=1)[:, np.newaxis]

    return similarity


def get_collision_centers( pos, size ):
    """
    Get the collision centers for the agent.   We are covering the vehicle with three circles, sized by the
    maximum of the width and one third the length of the vehicle.   The radius of each circle is sqrt(2) times
    the maximum of the width and 1/3 the length of the vehicle and centred on a 1/3 length block of the vehicle.
    """

    sep = size[0] / 3.0
    rad = sqrt(2) * max( sep, size[1] / 2.0 )
    cos_yaw = np.cos(pos[2])
    sin_yaw = np.sin(pos[2])
    points = np.array([ pos[:2] + np.array([ -sep*cos_yaw, -sep*sin_yaw ]),
                        pos[:2],
                        pos[:2] + np.array([ sep*cos_yaw, sep*sin_yaw ]) ])

    return points, rad

def get_min_distance( av_pos, av_size, agent_pos, agent_size ):
    """
    Get the minimum distance between the av and the agent
    """
    av_points, av_rad = get_collision_centers( av_pos, av_size )
    agent_points, agent_rad = get_collision_centers( agent_pos, agent_size )

    min_distance = np.inf
    for av_pt in av_points:
        for agent_pt in agent_points:
            dist = np.linalg.norm( av_pt - agent_pt )
            if dist < min_distance:
                min_distance = dist

    return min_distance - av_rad - agent_rad


def get_relative_velocity( av_velocity, agent_velocity ):
    """
    Get the relative velocity between the AV and the agent
    """
    return np.linalg.norm( av_velocity[:2] - agent_velocity[:2] )


def time_to_collision( av_pos, av_size, av_velocity, agents, prediction_num, dt=0.1 ):
    """
    Calculate the time to collision for the current trajectory with all of the listed agent
    trajectories.

    The TTC is found by finding the minimum distance between the two objects and dividing by the relative
    velocity of the two objects.  If the relative velocity is less than or equal to zero, then the TTC is
    set to infinity.

    Each agent is represented by three collision circles of radius root2*(max(length/3,width)), separated by
    length/3 along the length of the agent (left, centered, and right).

    The relative velocity is calculated by taking the difference of the two velocities and the difference of the
    two headings.  The relative velocity is then the magnitude of the relative velocity vector.

    """

    for agent in agents:
        min_distance = get_min_distance( av_pos, av_size, agent["predictions"][:, prediction_num, :3], agent["size"] )
        relative_velocity = get_relative_velocity( av_velocity, agent["predictions"][:, prediction_num, AgentTrack.DataColumn.DX:AgentTrack.DataColumm.DY+1] )

        ttc = min_distance / relative_velocity if relative_velocity > 0 else np.inf




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

    steps_per_prediction = int(prediction_iterval / dt)
    trajectory_steps = trajectory.shape[0]

    for agent in agents:

        prediction = agent.get_predictions()
        N, K, D = prediction.shape

        # initialize trajectory beliefs for each agent
        belief = np.zeros([ trajectory_steps, N])
        belief[0, :] = 1.0 / N

        # for each prediction
        for k in range(K):

            # We need to evaluate the probability of viewing each footprint of the agent at the current time step
            tic = time.time()
            visibility = {}
            for traj_num in range(N):
                visibility_dictionary = get_perception_dictionary(scenario=scenario, trajectory=trajectory,
                                                                  visibility=None, agents=agents, target_agent=agent,
                                                                  target_agent_trajectory=traj_num, prediction_num=k, beliefs=beliefs,
                                                                  prediction_interval=prediction_iterval, dt=dt )
                visibility[traj_num] = visibility_dictionary

            perception_time = time.time() - tic
            print(f"Perception time: {perception_time:.3f} seconds")

            similarity = calculate_similarity(agent, agent["predictions"], k)

            time_step = k * steps_per_prediction

            obs_loc = trajectory[time_step, :2]  # get the position of the AV at the current time step

            for candidate_num in range(N):

                belief[k, candidate_num] = 0
                for traj_num in range(N):
                    target_vis = visibility[traj_num][obs_loc]

                    occ = 0
                    for alt_idx in range(N):
                        alt_vis = visibility[alt_idx][obs_loc]
                        occ += (1 - alt_vis) * beliefs[k-1, alt_idx]

                    belief[k, candidate_num] += ( target_vis * similarity[candidate_num, traj_num] * belief[k-1, traj_num] +
                                                  (1 - target_vis) * occ )

            # normalize the belief
            belief[k, :] = belief[k, :] / np.sum(belief[k, :])


    # now evaluate each trajectory for the probability of stopping based on TTC > beta and occupancy/belief > alpha

    # return the mean probability of stopping over all agents and the min stopping time

