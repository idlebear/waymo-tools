#
# Track the state of an agent, watching their position, orientation, speed and
# acceration
#

import numpy as np
import pandas as pd

from trajectron.environment import Environment, Scene, Node
from trajectron.environment.data_utils import derivative_of

from enum import IntEnum


class AgentTrack:
    class DataColumm(IntEnum):
        X = 0
        Y = 1
        HEADING = 2
        TIME = 3

    def __init__(self, id, agent_type, pos_data=None, history_length=10, dt=1.0):
        """
        agent_type: trajectron.environment.NodeType
        """
        self.id = str(id)
        self.node = Node(node_type=agent_type, node_id=self.id, data=None)

        self.history_length = history_length
        self.dt = dt
        self.current_index = 0
        self.buffer_size = history_length * 10
        self.data = np.zeros((self.buffer_size, 4), dtype=object)  # x, y, t
        self.prediction = None

        if pos_data is not None:
            self.update(pos_data)

    def update(self, pos_data):
        """
        Update the agent's state with new position data.  Data must be continuous
        with any missing data filled in with NaN entries.

        pos_data: np.array of shape (n, 4), where n is the number of entries and
                    each entry is (x, y, orientation, t)

        """
        n = len(pos_data)

        for row in pos_data:
            x, y, heading, t = row

            while self.data[self.current_index - 1, AgentTrack.DataColumm.TIME] < t - 1:
                self._insert(np.nan, np.nan, np.nan, self.data[self.current_index - 1, AgentTrack.DataColumm.TIME] + 1)
            self._insert(x, y, heading, t)

        start_index = max(0, self.current_index - self.history_length)
        while np.isnan(self.data[start_index, AgentTrack.DataColumm.X]):
            start_index += 1

        x = self.data[start_index : self.current_index, AgentTrack.DataColumm.X].astype(float)
        y = self.data[start_index : self.current_index, AgentTrack.DataColumm.Y].astype(float)
        heading = self.data[start_index : self.current_index, AgentTrack.DataColumm.HEADING].astype(float)
        vx = derivative_of(x, self.dt)
        vy = derivative_of(y, self.dt)
        ax = derivative_of(vx, self.dt)
        ay = derivative_of(vy, self.dt)

        # data_dict = {
        #     ("position", "x"): x,
        #     ("position", "y"): y,
        #     ("velocity", "x"): vx,
        #     ("velocity", "y"): vy,
        #     ("acceleration", "x"): ax,
        #     ("acceleration", "y"): ay,
        # }
        if self.node.type == "VEHICLE":
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            a_norm = np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.0))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            header = [
                ("position", "x"),
                ("position", "y"),
                ("velocity", "x"),
                ("velocity", "y"),
                # ("velocity", "norm"),
                ("acceleration", "x"),
                ("acceleration", "y"),
                # ("acceleration", "norm"),
                # ("heading", "x"),
                # ("heading", "y"),
                ("heading", "°"),
                ("heading", "d°"),
            ]
            node_data = np.hstack(
                [
                    x.reshape(-1, 1),
                    y.reshape(-1, 1),
                    vx.reshape(-1, 1),
                    vy.reshape(-1, 1),
                    # v_norm,
                    ax.reshape(-1, 1),
                    ay.reshape(-1, 1),
                    # a_norm,
                    # heading_x.reshape(-1, 1),
                    # heading_y.reshape(-1, 1),
                    heading.reshape(-1, 1),
                    derivative_of(heading, self.dt, radian=True).reshape(-1, 1),
                ]
            )
        else:
            header = [
                ("position", "x"),
                ("position", "y"),
                ("velocity", "x"),
                ("velocity", "y"),
                ("acceleration", "x"),
                ("acceleration", "y"),
            ]
            node_data = np.hstack(
                [
                    x.reshape(-1, 1),
                    y.reshape(-1, 1),
                    vx.reshape(-1, 1),
                    vy.reshape(-1, 1),
                    ax.reshape(-1, 1),
                    ay.reshape(-1, 1),
                ]
            )
        self.node.overwrite_data(node_data, header)
        self.node.first_timestep = self.data[start_index, AgentTrack.DataColumm.TIME]

    def _insert(self, x, y, heading, t):
        if self.current_index >= self.history_length:
            preserved_len = max(0, self.history_length - 1)
            roll_back = max(0, self.current_index - preserved_len)
            self.data = np.roll(self.data, -roll_back, axis=0)
            self.current_index = preserved_len

        self.data[self.current_index] = [x, y, heading, t]
        self.current_index += 1

    def get(self, timestep, state):
        return self.node.get(np.array([timestep, timestep], dtype=int), state[self.node.type])

    def get_node(self):
        return self.node

    def set_prediction(self, prediction_start, prediction):
        self.prediction = prediction
        self.prediction_start = prediction_start

    def get_prediction(self, prediction_start):
        B, N, M, D = self.prediction.shape
        start_index = 0
        if prediction_start is not None:
            start_index = max(0, prediction_start - self.prediction_start)
        if self.prediction is None or start_index > M:
            return None

        return self.prediction[0, :, start_index : start_index + M, :]

    def plot_prediction(self, ax, colour=None, timestep=None):
        trajectories = self.get_prediction(timestep)
        if trajectories is not None:
            for trajectory in trajectories:
                ax.plot(trajectory[:, 0], trajectory[:, 1], label=f"agent_{self.id}")
