from math import ceil, floor, log, exp, sqrt, sin, cos, isnan, pi
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from threading import Lock

from config import GRID_SIZE, FIG_VISIBILITY_COSTMAP, DEBUG_VISIBILITY

"""
A simple class to store/manage the visibility cost map information -- we need to
make it persistant, and have it move with the vehicle (same logic as the Occupancy Grid)
"""


class VisibilityGrid:
    def __init__(self, dim, resolution=1, origin=(0.0, 0.0), alpha=1):
        self.dim = dim
        self.resolution = resolution
        self.origin = origin
        self.grid_size = int(dim / resolution)
        self.alpha = alpha

        # allocate a mutex/lock
        self.mutex = Lock()

        self.grid_fig = None

        self.reset(origin=origin)

    def reset(self, origin=(0.0, 0.0)):
        self.grid = np.zeros([self.grid_size, self.grid_size])
        self.origin = origin

    def copy(self):
        dup = VisibilityGrid(self.dim, self.resolution, (self.origin[0], self.origin[1]), self.invertY)
        dup.grid = np.array(self.grid)
        return dup

    def reframe(self, pos):
        return (int((pos[0] - self.origin[0]) / self.resolution), int((pos[1] - self.origin[1]) / self.resolution))

    def move_origin(self, new_origin):
        # calculate the offset that places the desired center in a block and
        # apply that shift to the stored value for the center.  The result is
        # a new center that is *close* to the desired value, but not exact
        # allowing the grid to keep the correct relative position of existing
        # occupancy
        dx = floor((new_origin[0] - self.origin[0]) / self.resolution)
        dy = floor((new_origin[1] - self.origin[1]) / self.resolution)

        if not dx and not dy:
            return

        self.origin = (
            self.origin[0] + dx * self.resolution,
            self.origin[1] + dy * self.resolution,
        )

        if dx > 0:
            old_x_min = min(self.grid_size, dx)
            old_x_max = self.grid_size
            new_x_min = 0
            new_x_max = max(0, self.grid_size - dx)
        else:
            old_x_min = 0
            old_x_max = max(0, self.grid_size + dx)
            new_x_min = min(self.grid_size, -dx)
            new_x_max = self.grid_size

        if dy > 0:
            old_y_min = min(self.grid_size, dy)
            old_y_max = self.grid_size
            new_y_min = 0
            new_y_max = max(0, self.grid_size - dy)
        else:
            old_y_min = 0
            old_y_max = max(0, self.grid_size + dy)
            new_y_min = min(self.grid_size, -dy)
            new_y_max = self.grid_size

        tmp_grid = np.zeros_like(self.grid)
        if old_x_max - old_x_min > 0 and old_y_max - old_y_min > 0:
            tmp_grid[new_y_min:new_y_max, new_x_min:new_x_max] = self.grid[old_y_min:old_y_max, old_x_min:old_x_max]
        self.grid = tmp_grid

    def update(self, points, values):
        self.mutex.acquire()
        try:
            # apply forgetfulnes to the entire map first.  If alpha is positive, then we are applying a proportional update.  We
            # have to update the map first since the cell update is only for a subset of the grid.
            if self.alpha >= 0:
                self._decay(1.0 - self.alpha)
            else:
                self._decay(abs(self.alpha))

            # add in the visibility values
            for pt, val in zip(points, values):
                if self.alpha < 0:
                    # use the max method
                    self.grid[pt[1], pt[0]] = max(val, self.grid[pt[1], pt[0]])
                else:
                    # the entire grid has already been decayed by (1-alpha)
                    self.grid[pt[1], pt[0]] = self.alpha * val + self.grid[pt[1], pt[0]]

            if DEBUG_VISIBILITY:
                self.__visualize()

        finally:
            self.mutex.release()

    def visibility_costmap(self):
        self.mutex.acquire()
        try:
            return np.array(self.grid)
        finally:
            self.mutex.release()

    def value(self, x, y):
        (x, y) = self.reframe((x, y))
        return self.grid[y, x]

    def decay(self, rate):
        self.mutex.acquire()
        try:
            return self._decay(rate)
        finally:
            self.mutex.release()

    def _decay(self, rate):
        self.grid *= rate

    def __visualize(self):
        if self.grid_fig is None:
            self.grid_fig, self.grid_ax = plt.subplots(num=FIG_VISIBILITY_COSTMAP)
            self.grid_img = self.grid_ax.imshow(np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8))
            plt.show(block=False)

        img = Image.fromarray(((1.0 - np.flipud(self.grid)) * 255.0).astype(np.uint8)).convert("RGB")
        self.grid_img.set_data(img)

        # self.grid_fig.canvas.draw()
        # self.grid_fig.canvas.flush_events()
