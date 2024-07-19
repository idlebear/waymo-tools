from trajectory_planner.frenet_optimal_trajectory import (
    generate_target_course,
    frenet_optimal_planning,
    PlannerArgs,
    Frenet_path,
)
import numpy as np

DEFAULT_PLANNING_HORIZON = 10
TICK = 0.1


class TrajectoryPlanner:
    def __init__(self, data: np.ndarray) -> None:
        self._t = data[:, 0]
        self._waypoints = data[:, 2:4]
        self._yaw = data[:, 4]

        self._distances = np.linalg.norm(np.diff(self._waypoints, axis=0), axis=1)
        self._s = np.cumsum(self._distances)
        times = np.diff(self._t) / 1.0e6
        self._v = self._distances / times
        self._dt = np.mean(times)

        self._tx, self._ty, self.yaw, self.curvature, self._csp = generate_target_course(self._waypoints)

    def generate_trajectories(
        self, trajectories_requested: int = 1, planning_horizon: int = DEFAULT_PLANNING_HORIZON, step: int = 0
    ) -> list:

        if step + planning_horizon >= len(self._t) - 1:
            raise ValueError("Step + Planning Horizon extends beyond available data")

        planner_args = PlannerArgs(
            min_predict_time=planning_horizon * self._dt,
            max_predict_time=planning_horizon * self._dt,
            predict_step=self._dt,
            time_tick=self._dt,
            target_speed=self._v[step + planning_horizon],
            stopping_time=None,  # EMERGENCY_STOPPING_TIME,
            trajectories_requested=trajectories_requested,
            generate_planning_path=True,
        )

        self._trajectories = frenet_optimal_planning(
            self._csp,
            self._s[step],
            self._v[step],
            0,  # self._c_d,
            0,  # self._c_d_d,
            0,  # self._c_d_dd,
            0,  # acceleration
            planner_args,
        )

        return self.get_working_trajectories()

    def get_working_trajectories(self) -> list:
        return self._trajectories[1]

    def get_planning_trajectory(self) -> Frenet_path:
        return self._trajectories[-1][0]
