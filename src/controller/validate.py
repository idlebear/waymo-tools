from config import *
import numpy as np
from skimage.morphology import binary_dilation, binary_opening
import matplotlib.pyplot as plt

TRAJECTORIES_TO_VISUALIZE = 100


def create_2d_gaussian(width, centre_x=0, centre_y=0, sigma=1, scale=1):
    """create_2d_gaussian -- creates a 2D gaussian kernal at the
           specified offset, and scale

    @param width     the number of steps in the gaussian kernel
    @param offset_x  the distance from center of the peak value in X.
    @param offset_y  the distance from center of the peak value in Y.
    @param std       sigma -- the standard deviation of the curve
    @param scale     numerical width of each step
    """
    X = range(width)
    x, y = np.meshgrid(X, X)

    xc = int(width / 2.0 - scale * centre_x)
    yc = int(width / 2.0 - scale * centre_y)

    sigma = sigma * scale

    # since the gaussian may be off-center (and is definitely truncated), normalize
    # using the sum of elements
    gus = np.exp(-(np.power(x - xc, 2) + np.power(y - yc, 2)) / (2 * sigma * sigma))
    return gus / np.sum(gus)


# Basic step function -- apply the control to advance one step
def euler(vehicle, state, control):
    return vehicle.ode(state, control)


#
# Also define the Runge-Kutta variant as it is (apparently) a much
# better approximation of the first order derivative
#
# https://en.wikipedia.org/wiki/Runge-Kutta_methods
def runge_kutta_step(vehicle, state, control, dt):
    k1 = vehicle.ode(state, control)
    k2 = vehicle.ode(state + k1 * (dt / 2), control)
    k3 = vehicle.ode(state + k2 * (dt / 2), control)
    k4 = vehicle.ode(state + k3 * dt, control)

    return (k1 + 2 * (k2 + k3) + k4) / 6.0


# wrapper to allow easy switch between methods. (simplifies validation)
def step_fn(vehicle, state, control, dt=None):
    # return euler(vehicle=vehicle, state=state, control=control)
    return runge_kutta_step(vehicle=vehicle, state=state, control=control, dt=dt)


def run_trajectory(vehicle, initial_state, controls, dt):

    traj = np.zeros((len(controls) + 1, len(initial_state)))
    traj[0, :] = initial_state

    state = np.array(initial_state)
    for m, u in enumerate(controls):
        step = step_fn(vehicle=vehicle, state=state, control=u, dt=dt)
        state += step * dt
        traj[m + 1, :] = state

    return traj


def visualize_controls(plot_data, vehicle, initial_state, nom_traj, controls, dt):
    # visualizing!

    colours = {
        "ours": "blue",
        "andersen": "red",
        "higgins": "green",
        "none": "orange",
        "ignore": "pink",
    }

    new_plot = False
    if plot_data is None:
        new_plot = True
        fig, ax = plt.subplots(num=FIG_MPC)

        plot_data = {
            "figure": fig,
            "ax": ax,
            "plot_backgrounds": [],
        }
    else:
        fig = plot_data["figure"]
        ax = plot_data["ax"]

    try:
        plot_data["nom_line"][0].set_data(nom_traj[:, 0], nom_traj[:, 1])
    except KeyError:
        new_plot = True
        plot_data["nom_line"] = ax.plot(nom_traj[:, 0], nom_traj[:, 1], color="black", linestyle="--")

    for key, value in controls:
        traj = run_trajectory(vehicle=vehicle, initial_state=initial_state, controls=value, dt=dt)
        try:
            plot_data[key][0].set_data(traj[:, 0], traj[:, 1])
        except KeyError:
            plot_data[key] = ax.plot(traj[:, 0], traj[:, 1], color=colours[key])

    if new_plot:
        ax.axis("equal")
        plt.show(block=False)
    else:
        ax.relim()
        ax.autoscale_view()

    plt.pause(0.001)

    return plot_data


def visualize_variations(plot_data, vehicle, initial_state, nom_traj, u_nom, u_variations, u_weighted, dt):
    # visualizing!

    if plot_data is None:
        fig, ax = plt.subplots(num=FIG_MPC, nrows=1, ncols=2)
        plot_data = {
            "figure": fig,
            "ax": ax,
            "plot_lines": None,
            "weighted_line": None,
            "nom_line": None,
            "plot_backgrounds": [],
        }
        plot_lines = None

    else:
        fig = plot_data["figure"]
        ax = plot_data["ax"]
        plot_lines = plot_data["plot_lines"]

    n_samples, n_controls, n_steps = u_variations.shape
    indexes = np.random.choice(n_samples, min(n_samples, TRAJECTORIES_TO_VISUALIZE), replace=False)

    new_traj_pts = []
    for i in indexes:
        u_var = np.array(u_nom)
        u_var = u_var + u_variations[i, ...].T

        traj = run_trajectory(vehicle=vehicle, initial_state=initial_state, controls=u_var, dt=dt)
        new_traj_pts.append(np.expand_dims(traj, axis=0))

    new_traj_pts = np.vstack(new_traj_pts)

    traj = run_trajectory(vehicle=vehicle, initial_state=initial_state, controls=u_weighted, dt=dt)

    # nom_traj = run_trajectory(vehicle=vehicle, initial_state=initial_state, controls=u_nom, dt=dt)

    if plot_lines is None:
        plot_data["plot_lines"] = ax[0].plot(new_traj_pts[:, :, 0].T, new_traj_pts[:, :, 1].T)
        plot_data["nom_line"] = ax[1].plot(nom_traj[:, 0], nom_traj[:, 1], color="red")
        plot_data["weighted_line"] = ax[1].plot(traj[:, 0], traj[:, 1], color="blue")

        ax[0].axis("equal")
        ax[1].axis("equal")
        plt.show(block=False)
    else:
        for line, data in zip(plot_data["plot_lines"], new_traj_pts):
            line.set_data(data[:, 0], data[:, 1])

        plot_data["weighted_line"][0].set_data(traj[:, 0], traj[:, 1])
        plot_data["nom_line"][0].set_data(nom_traj[:, 0], nom_traj[:, 1])

        ax[0].relim()
        ax[0].autoscale_view()
        ax[1].relim()
        ax[1].autoscale_view()

        plt.pause(0.001)

    return plot_data


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


def validate_controls(vehicle, initial_state, controls, obs, static_objects, resolution, dt) -> np.array:
    N_controls = len(controls)

    # calculate the future states
    states = run_trajectory(vehicle=vehicle, initial_state=initial_state, controls=controls, dt=dt)

    # remove all the static structures/objects  - blot any inaccuracies as well
    # construct a mask of static objects from the map based on the object's position and size
    static_mask = np.zeros_like(obs)
    for agent in static_objects:
        draw_agent(
            map=static_mask,
            origin=initial_state,
            resolution=GRID_CELL_WIDTH,
            centre=agent["centre"],
            size=agent["size"],
            yaw=agent["yaw"],
            visibility=1.0,
        )

    pedestrian_grid = np.where(static_mask == 1, 0, obs)
    pedestrian_grid = np.where(pedestrian_grid >= 0.9, 0, pedestrian_grid)
    pedestrian_grid = np.where(pedestrian_grid >= 0.4, 1, 0)

    max_pedestrian_move = MAX_PEDESTRIAN_SPEED * dt / resolution
    steps_between_dilation = max(1, np.round(1.0 / max_pedestrian_move))
    max_pedestrian_move = int(np.ceil(max_pedestrian_move))

    pedestrian_grid = binary_opening(pedestrian_grid, footprint=np.ones((3, 3)))

    # construct a stack of maps showing possible future occupancy (inflated)
    future_maps = []
    for i in range(N_controls):
        # blur at the start of each interval, assuming worst case
        if i % steps_between_dilation == 0:
            dilation = binary_dilation(pedestrian_grid, footprint=np.ones((3, 3)))

            # blurred_grid = np.clip(convolve2d(in1=pedestrian_grid, in2=blur, mode="same"), 0, 1)
            # next_ped_grid = np.maximum(blurred_grid, pedestrian_grid)
            pedestrian_grid = np.where(static_mask == 1, 0, dilation)
        future_maps.append(pedestrian_grid)

    # check the projected states (skipping the first -- that has to be ok)
    initial_state = states[0]
    future_states = states[1:]
    for state_index, state in enumerate(future_states):
        x = int((state[0] - initial_state[0]) / resolution + GRID_SIZE // 2)
        y = int((state[1] - initial_state[1]) / resolution + GRID_SIZE // 2)
        if future_maps[state_index][y, x] > OCCUPANCY_THRESHOLD:
            # potential collision
            break
    if state_index >= N_controls:
        # no collisions detected
        return controls[0, :]

    # current state has a possible collision -- assume that we must be able to stop in the previous
    # step at state current_state -1
    state_index -= 1
    max_v = abs(vehicle.min_a) * dt * (state_index - 1)

    # while current_state > 0:
    #     current_state -= 1
    #     max_v = max_v + abs(vehicle.min_a) * dt
    #     if future_states[current_state][2] < max_v:
    #         # we're within stopping acceleration, should be safe to proceed.
    #         return controls[0, :]

    # fell all the way back to the initial state -- calculate the accleration
    # required to bring the car back to spec.
    dv = max_v - initial_state[2]
    required_a = np.round(dv / dt, 3)

    # now check if it's valid
    requested_a = controls[0, 0]
    if required_a < vehicle.min_a:
        # emergency braking required
        print("EMERGENCY brake deployed!")
        return np.array([vehicle.min_a, controls[0, 1]])
    elif required_a < requested_a:
        print("Slowing...!")
        return np.array([required_a, controls[0, 1]])

    # all good
    return controls[0, :]
