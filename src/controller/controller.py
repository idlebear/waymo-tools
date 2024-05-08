import numpy as np
from numpy import pi as PI
from math import sqrt

from time import time

from config import *
from controller.ModelParameters.Ackermann import Ackermann4
from controller.mppi_gpu import MPPI
from controller.validate import visualize_variations, validate_controls, draw_agent

X_WEIGHT = 250.0
Y_WEIGHT = 250.0
V_WEIGHT = 150.0
THETA_WEIGHT = 10.0
A_WEIGHT = 100
DELTA_WEIGHT = 50

DEFAULT_ACCELERATION = 4.0
CONTROL_VARIATION_LIMITS = [5, PI / 6.0]
CONTROL_LIMITS = [5, PI / 5.0]

DEFAULT_METHOD_WEIGHT = 100.0
DEFAULT_LAMBDA = 50000.0


def init_controller(args):
    vehicle_model = Ackermann4()
    Q = np.array([args.x_weight, args.y_weight, args.v_weight, args.theta_weight])
    R = np.array([args.a_weight, args.delta_weight])
    controller = MPPI(
        vehicle=vehicle_model,
        samples=args.samples,
        seed=args.seed,
        u_limits=CONTROL_LIMITS,
        u_dist_limits=CONTROL_VARIATION_LIMITS,
        c_lambda=args.c_lambda,
        Q=Q,
        R=R,
        M=args.mppi_m,
        method=args.method,
        scan_range=LIDAR_RANGE,
        discount_factor=args.discount,
    )

    return controller


def __get_bb_vertices(agent):
    cx, cy = agent["centre"][:2]
    dx, dy = agent["size"][:2]
    yaw = agent["yaw"]

    pts = [  # top left, top right, bottom right, bottom left
        [-dx / 2, dy / 2],
        [dx / 2, dy / 2],
        [dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
    ]

    # rotate the points
    for i in range(len(pts)):
        x = pts[i][0]
        y = pts[i][1]
        pts[i][0] = x * np.cos(yaw) - y * np.sin(yaw) + cx
        pts[i][1] = x * np.sin(yaw) + y * np.cos(yaw) + cy

    return pts


def get_control(
    controller,
    costmap,
    occupancy,
    u_nom,
    trajectory,
    target_speed,
    planning_horizon=10,
    dt=0.1,
    v=0.0,
    pos=[0.0, 0.0],
    yaw=0,
    visible_agents={},
    method="Ignore",
):

    if u_nom is None or np.any(np.isnan(u_nom)):
        if u_nom is not None and np.any(np.isnan(u_nom)):
            print("WARN: NaN in nominal control")
        u_nom = np.zeros((planning_horizon, 2))
        update_start = 0
    else:
        u_nom[0:-1, :] = u_nom[1:, :]
        update_start = planning_horizon - 1

    v_est = v
    for i in range(update_start, planning_horizon):
        dtheta = (trajectory.yaw[i + 1] - trajectory.yaw[i]) / dt
        dv = trajectory.s_d[i + 1] - trajectory.s_d[i]
        v_avg = (trajectory.s_d[i + 1] + trajectory.s_d[i]) / 2.0

        if v_est < target_speed:
            u_nom[i, 0] = DEFAULT_ACCELERATION
        else:
            u_nom[i, 0] = dv / dt
        v_est = v_est + DEFAULT_ACCELERATION * dt

        # self.u_nom[i, 0] = DEFAULT_ACCELERATION # dv / self._tick
        u_nom[i, 1] = np.nan_to_num(np.arctan((dtheta) * controller.vehicle.L / v_avg))
        if u_nom[i, 1] > CONTROL_LIMITS[1]:
            u_nom[i, 1] = CONTROL_LIMITS[1]
        elif u_nom[i, 1] < -CONTROL_LIMITS[1]:
            u_nom[i, 1] = -CONTROL_LIMITS[1]

    x_nom = np.zeros((planning_horizon + 1, 4))
    pts_to_update = min(planning_horizon + 1, len(trajectory.x))
    x_nom[:pts_to_update, 0] = trajectory.x[:pts_to_update]
    x_nom[:pts_to_update, 1] = trajectory.y[:pts_to_update]
    x_nom[:pts_to_update, 2] = trajectory.s_d[:pts_to_update]
    x_nom[:pts_to_update, 3] = trajectory.yaw[:pts_to_update]
    if pts_to_update < x_nom.shape[0]:
        x_nom[pts_to_update:, :] = x_nom[pts_to_update - 1, :][np.newaxis, :]

    origin = [
        pos[0] - (GRID_SIZE / 2) * GRID_CELL_WIDTH + GRID_CELL_WIDTH / 2.0,
        pos[1] - (GRID_SIZE / 2) * GRID_CELL_WIDTH + GRID_CELL_WIDTH / 2.0,
    ]

    initial_state = [pos[0], pos[1], v, yaw]

    actors = []
    distances = []
    if method != "Ignore":
        for agent in visible_agents:
            # for each agent, collect the center, bounding box, radius, closest point, and distance from the ego vehicle
            # BUGBUG -- this is a bit of a hack -- we're assuming the bounding box is a circle based on the width of the vehicle
            #           and not the worst case length.  We need to represent agents as a series circles, but for now this should
            #           be good enough.
            extent = 1.0  # min(agent["size"][0:2]) / 2.0
            x, y = agent["centre"][:2]
            distance = sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
            distances.append(distance)

            # TODO: when implementing tracking for the closest approach, uncomment this code
            # try:
            #     if self.object_distances[agent_data["id"]] > distance:
            #         self.object_distances[agent_data["id"]] = distance
            #         print(f"    New distance: {distance} for {agent_data['id']} ")
            # except KeyError:
            #     self.object_distances[agent_data["id"]] = distance
            #     print(f"    New distance: {distance} for {agent_data['id']} ")

            # Calculate the smallest angle to the bounding box -- used in the Andersen cost function
            vertices = __get_bb_vertices(agent)
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
    else:
        # just check the distances
        for agent in visible_agents:
            x, y = agent["centre"][:2]
            distance = sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
            distances.append(distance)
            # try:
            #     if self.object_distances[agent_data["id"]] > distance:
            #         self.object_distances[agent_data["id"]] = distance
            #         print(f"    New distance: {distance} for {agent_data['id']} ")
            # except KeyError:
            #     self.object_distances[agent_data["id"]] = distance
            #     print(f"    New distance: {distance} for {agent_data['id']} ")

    actors = np.array(actors)

    tic = time()
    u, u_var = controller.find_control(
        costmap=costmap.visibility_costmap(),
        origin=origin,
        resolution=GRID_CELL_WIDTH,
        x_init=initial_state,
        x_nom=x_nom,
        u_nom=u_nom,
        actors=actors,
        dt=dt,
    )
    mppi_time = time() - tic

    # print(f"Control: {u[0,:]}, Speed: {v}")

    # failed_trajectories = np.sum(u[0, :] == 0)
    # if failed_trajectories > samples / 2 or np.any(np.isnan(u[0, :])):
    #     print("WARN: Failed to find a valid trajectory.  Emergency braking")
    #     u *= 0
    #     next_control = u[0, :]
    #     return

    # if method != "Ignore":
    #     tic = time()
    #     next_control = validate_controls(
    #         vehicle=controller.vehicle,
    #         initial_state=initial_state,
    #         controls=u,
    #         obs=occupancy,
    #         static_objects=visible_agents,
    #         resolution=GRID_CELL_WIDTH,
    #         dt=dt,
    #     )
    #     print(f"    Validation time: {time() - tic}")
    # else:
    #     next_control = u[0, :]

    return u, u_var
