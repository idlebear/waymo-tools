from enum import Enum
from math import floor, ceil, log
import numpy as np
from random import random
from scipy import signal

# Clamp probability to maximum and minimum values to prevent overflow and subsequent
# irrational behaviour -- limiting the values closer to zero makes the occupancy grid more
# responsive to dynamic objects in the environment
MIN_PROBABILITY = -60.0
MAX_PROBABILITY = 60.0

MAP_INCREMENT = 100


class Grid:
    PROBABILITY_THRESHOLD = 0.0000001

    def __init__(self, origin, height, width, resolution, initial_value=0):
        self._origin = np.array([origin[0], origin[1]])
        self._resolution = resolution
        self._width = width
        self._height = height
        self._initial_value = initial_value
        self._grid = np.ones((height, width)) * self._initial_value

    def reframe(self, pos):
        return ((pos - self._origin) / self._resolution).astype(int)

    def move_origin(self, dest, move_data=True):
        # calculate the offset that places the desired center in a block and
        # apply that shift to the stored value for the center.  The result is
        # a new center that is *close* to the desired value, but not exact
        # allowing the grid to keep the correct relative position of existing
        # occupancy
        dx = floor((dest[0] - self._origin[0]) / self._resolution)
        dy = floor((dest[1] - self._origin[1]) / self._resolution)

        if not dx and not dy:
            return

        self._origin = self._origin + np.array([dx, dy]) * self._resolution

        if move_data is True:
            if dx > 0:
                old_x_min = dx
                old_x_max = self._width
                new_x_min = 0
                new_x_max = self._width - dx
            else:
                old_x_min = 0
                old_x_max = self._width + dx
                new_x_min = -dx
                new_x_max = self._width

            if dy > 0:
                old_y_min = dy
                old_y_max = self._height
                new_y_min = 0
                new_y_max = self._height - dy
            else:
                old_y_min = 0
                old_y_max = self._height + dy
                new_y_min = -dy
                new_y_max = self._height

            tmp_grid = np.ones_like(self._grid) * self._initial_value
            tmp_grid[new_x_min:new_x_max, new_y_min:new_y_max] = self._grid[old_x_min:old_x_max, old_y_min:old_y_max]
            self._grid = tmp_grid
        else:
            self._grid[:, :] = self._initial_value

    def get_value_at(self, x, y):
        x, y = self.reframe(x, y)
        if x < 0 or x >= self._width or y < 0 or y >= self._height:
            raise IndexError("Input vertex ({},{}) outside of map area".format(x, y))

        return np.array(self._grid[x, y])

    def get_grid(self):
        # return a copy of the probability grid, ready for manipulation
        return np.array(self._grid)


class ProbabilityGrid(Grid):
    def __init__(self, origin, height, width, resolution, initial_prob=0.5) -> None:
        self.l0 = log(initial_prob / (1 - initial_prob))
        super().__init__(origin, height, width, resolution, self.l0)

    def reset_probability(self, prob=0):
        # set the internal probability representation to the supplied value
        self._grid[:, :] = prob

    def __expand_map(self, x, y, ex, ey):
        if x < 0 or y < 0 or ex >= self._width or ey >= self._height:
            new_width = self._width
            new_height = self._height

            old_x_origin, old_y_origin = 0, 0
            while x < 0:
                new_width = new_width + MAP_INCREMENT
                self._origin[0] -= MAP_INCREMENT * self._resolution
                x += MAP_INCREMENT
                old_x_origin += MAP_INCREMENT
            while y < 0:
                new_height = new_height + MAP_INCREMENT
                self._origin[1] -= MAP_INCREMENT * self._resolution
                y += MAP_INCREMENT
                old_y_origin += MAP_INCREMENT
            while ex >= new_width:
                new_width += MAP_INCREMENT
            while ey >= new_height:
                new_height += MAP_INCREMENT

            new_grid = np.ones((new_height, new_width)) * self.l0
            new_grid[
                old_y_origin : old_y_origin + self._height,
                old_x_origin : old_x_origin + self._width,
            ] = self._grid

            self._grid = new_grid
            self._width = new_width
            self._height = new_height

        return x, y

    def set_probability_at(self, pos, prob, replace=False):
        x, y = self.reframe(pos)
        dy, dx = prob.shape
        ey = y + dy
        ex = x + dx

        x, y = self.__expand_map(x, y, ex, ey)

        prob = np.log(prob / (1.0 - prob))
        if replace:
            self._grid[y : y + dy, x : x + dx] = prob
        else:
            self._grid[y : y + dy, x : x + dx] = np.clip(
                self._grid[y : y + dy, x : x + dx] - self.l0 + prob,
                MIN_PROBABILITY,
                MAX_PROBABILITY,
            )

    def get_probability_at(self, pos, shape):
        x, y = self.reframe(pos)
        off_x, off_y = 0, 0
        dy, dx = shape
        if x < 0:
            off_x = -x
            dx += x
            x = 0
        if y < 0:
            off_y = -y
            dy += y
            y = 0
        if x + dx > self._width:
            dx = self._width - x
        if y + dy > self._height:
            dy = self._height - y

        prob = np.ones(shape) * self.l0
        prob[off_y : off_y + dy, off_x : off_x + dx] = self._grid[y : y + dy, x : x + dx]

        num = np.zeros(shape)
        denom = np.ones(shape)
        np.exp(prob, out=num)
        denom = denom + num
        return np.divide(num, denom)

    def set_materials_at(self, pos, materials):
        x, y = self.reframe(pos)

        self.__expand_map(x, y, x, y)

        for key, value in materials.items():
            dx, dy = material.shape

        prob = np.log(materials / (1.0 - materials))
        self._grid[y : y + dy, x : x + dx] = prob

    def probability_map(self):
        # return the current state as a probabilistic representation
        num = np.zeros([self._height, self._width])
        denom = np.ones([self._height, self._width])
        np.exp(self._grid, out=num)
        denom = denom + num
        return np.divide(num, denom)

    def graph_probability(self, ax=None):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            (
                ly,
                lx,
            ) = self._grid.shape
            xv, yv = np.meshgrid(range(lx), range(ly), indexing="xy")
            ax.plot_wireframe(yv, xv, self.probability_map())
            ax.view_init(30, -120)
            plt.xlabel("X")
            plt.show()
            plt.close(fig)


# class MaterialsGrid(Grid):
#     def __init__(self, origin, height, width, resolution) -> None:
#         super().__init__(origin, height, width, resolution)
#         self.

#     def set_material_at(self, pos, material):
#         x, y = self.reframe(pos)
#         self._grid[y, x] = material


if __name__ == "__main__":
    grid = ProbabilityGrid((0, 0), 100, 100, 0.5, initial_prob=0.75)

    grid.set_probability_at(np.array((-3, 2)), np.ones((10, 10)) * 0.3)
    grid.set_probability_at(np.array((10, 10)), np.ones((10, 10)) * 0.8)
    grid.set_probability_at(np.array((70, 80)), np.ones((10, 10)) * 0.8)

    grid.graph_probability()
