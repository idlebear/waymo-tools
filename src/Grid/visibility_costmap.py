#
# Planning/Policy for visibility
#
from scipy import ndimage
import numpy as np
from time import time

from config import *
from Grid.polygpu import visibility_from_region

from Grid.VisibilityGrid import VisibilityGrid


def dump_grid(grid):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(grid)
    plt.show(block=False)


def update_visibility_costmap(costmap, map, centre, obs_trajectory, lane_width, target_pts):
    # move the costmap to reflect the updated position of the car - origin is the bottom left corner of the grid
    origin = [
        centre[0] - (GRID_SIZE / 2) * GRID_CELL_WIDTH + GRID_CELL_WIDTH / 2.0,
        centre[1] - (GRID_SIZE / 2) * GRID_CELL_WIDTH + GRID_CELL_WIDTH / 2.0,
    ]
    costmap.move_origin(origin)

    distance_grid = np.ones([GRID_SIZE, GRID_SIZE])

    x0 = origin[0]
    y0 = origin[1]

    locs = [
        [
            int((x - x0) / GRID_CELL_WIDTH),
            int((y - y0) / GRID_CELL_WIDTH),
        ]
        for (x, y) in zip(obs_trajectory.x, obs_trajectory.y)
    ]
    for loc in locs:
        if loc[0] < GRID_SIZE and loc[1] < GRID_SIZE:
            distance_grid[loc[1], loc[0]] = 0
    distance_grid = ndimage.distance_transform_cdt(distance_grid, metric="chessboard").astype(np.float64)

    obs_pts = np.where(distance_grid < (lane_width) / GRID_CELL_WIDTH)

    # place the obs points in the larger map
    obs_pts = [[x + GRID_SIZE // 2, y + GRID_SIZE // 2] for y, x in zip(*obs_pts)]

    def dump_targets(map, target_pts, results):
        grid = np.array(map)

        for (x, y), val in zip(target_pts, results):
            grid[y, x] = val

        dump_grid(grid)

    if len(target_pts):
        # results are num observation points rows by num region of interest points columns
        result = np.zeros((len(obs_pts), len(target_pts)))
        result = visibility_from_region(map, obs_pts, target_pts).reshape((len(obs_pts), -1))
        # log_result = -np.log(result + 0.00000001) * result
        # summed_log_result = np.sum(log_result, axis=1)

        def draw_vis(map, pts, src, result):
            visibility_map = np.array(map)
            for pt, val in zip(pts, result):
                visibility_map[pt[1], pt[0]] = max(val / 2 + 0.1, visibility_map[pt[1], pt[0]])
            visibility_map[src[1], src[0]] = 2

            dump_grid(visibility_map)

        summed_result = np.sum(result, axis=1)

        # to normalize the results, convert the visibility into a proportion of the region requested and convert to
        # non-info so we penalize locations that give nothing
        # summed_result /= len(target_pts)
        # assert np.max(summed_result) <= 1
        min_sum = np.min(summed_result)
        summed_result -= min_sum
        max_sum = np.max(summed_result)
        if max_sum:
            summed_result /= max_sum

        for pt in obs_pts:
            pt[0] = int(pt[0] - GRID_SIZE // 2)
            pt[1] = int(pt[1] - GRID_SIZE // 2)

        costmap.update(obs_pts, summed_result)

    return obs_pts
