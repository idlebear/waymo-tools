"""
The MIT License (MIT)

Copyright (c) 2016 - 2022 Atsushi Sakai and other contributors:
https://github.com/AtsushiSakai/PythonRobotics/contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame](https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math

try:
    from trajectory_planner import cubic_spline_planner
except:
    import cubic_spline_planner


class PlannerArgs:
    def __init__(
        self,
        max_speed=50.0 / 3.6,
        max_accel=100.0,
        max_curvature=1,
        max_road_width=2,
        d_road_w=0.4,
        time_tick=0.1,
        max_predict_time=5.2,
        min_predict_time=5,
        predict_step=0.1,
        target_speed=30 / 3.6,
        D_T_S=3.0 / 3.6,
        sampling_num_target_speed=20,
        stopping_time=2,
        max_brake_acceleration=3,
        trajectories_requested=1,
        generate_planning_path=True,
    ):
        self.MAX_SPEED = max_speed  # maximum speed [m/s]
        self.MAX_ACCEL = max_accel  # maximum acceleration [m/ss]
        self.MAX_CURVATURE = max_curvature  # maximum curvature [1/m]
        self.MAX_ROAD_WIDTH = max_road_width  # maximum road width [m]
        self.D_ROAD_W = d_road_w  # road width sampling length [m]
        self.DT = time_tick  # time tick [s]
        self.MAXT = max_predict_time  # max prediction time [m]
        self.MINT = min_predict_time  # min prediction time [m]
        self.PREDICT_STEP = predict_step
        self.TARGET_SPEED = target_speed  # target speed [m/s]
        self.D_T_S = D_T_S  # target speed sampling length [m/s]
        self.N_S_SAMPLE = sampling_num_target_speed  # sampling number of target speed
        self.MAX_BRAKING_ACCELERATION = (
            max_brake_acceleration  # maximum braking acceleration for emergency stops (planned)
        )
        self.GENERATE_PLANNING_PATH = generate_planning_path
        self.TRAJECTORIES_REQUESTED = trajectories_requested

        # cost weights
        self.KJ = 0.1
        self.KT = 0.1
        self.KD = 1.0
        self.KLAT = 1.0
        self.KLON = 1.0

        # stopping trajectory time
        self.STOPPING_TIME = stopping_time


ROBOT_RADIUS = 2.0  # robot radius [m]


class quintic_polynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T**3, T**4, T**5], [3 * T**2, 4 * T**3, 5 * T**4], [6 * T, 12 * T**2, 20 * T**3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2, vxe - self.a1 - 2 * self.a2 * T, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt


class quartic_polynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, T):
        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T**2, 4 * T**3], [6 * T, 12 * T**2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class Frenet_path:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        self.stopping = False


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, a, s0, planner_args):
    frenet_paths = []

    MAX_ROAD_WIDTH = planner_args.MAX_ROAD_WIDTH
    MINT = planner_args.MINT
    MAXT = planner_args.MAXT
    DT = planner_args.DT
    TARGET_SPEED = planner_args.TARGET_SPEED
    D_T_S = planner_args.D_T_S

    KJ = planner_args.KJ
    KT = planner_args.KT
    KD = planner_args.KD
    KLON = planner_args.KLON
    KLAT = planner_args.KLAT

    offset = 0  # centre only
    time_list = [MINT]

    if planner_args.STOPPING_TIME is not None:
        # generate  a safety path

        # Lateral motion planning
        fp = Frenet_path()
        lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, c_d, 0.0, 0.0, planner_args.STOPPING_TIME)

        fp.t = [t for t in np.arange(0.0, planner_args.STOPPING_TIME, DT)]
        fp.d = [lat_qp.calc_point(t) for t in fp.t]
        # if abs(fp.d[1]) > MAX_ROAD_WIDTH + 1:
        #     # TODO: small hack to catch badly generate trajectories
        #     raise SystemError("ERROR: Generated trajectories are badly formed")
        fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
        fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
        fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

        lon_qp = quartic_polynomial(
            s0, c_speed, -planner_args.MAX_BRAKING_ACCELERATION, 0, 0.0, planner_args.STOPPING_TIME
        )

        fp.s = [lon_qp.calc_point(t) for t in fp.t]
        fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
        fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
        fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

        Jp = sum(np.power(fp.d_ddd, 2))  # square of jerk
        Js = sum(np.power(fp.s_ddd, 2))  # square of jerk

        # square of diff from target speed
        ds = (TARGET_SPEED - fp.s_d[-1]) ** 2

        fp.cd = KJ * Jp + KT * planner_args.STOPPING_TIME + KD * fp.d[-1] ** 2
        fp.cv = KJ * Js + KT * planner_args.STOPPING_TIME + KD * ds
        fp.cf = KLAT * fp.cd + KLON * fp.cv
        fp.stopping = True
        safety_paths = [
            fp,
        ]
    else:
        safety_paths = []  # no safety path
    frenet_paths.append(safety_paths)

    d_steps = [0.0]
    if planner_args.TRAJECTORIES_REQUESTED > 1:
        d_step = MAX_ROAD_WIDTH / (2 * (planner_args.TRAJECTORIES_REQUESTED - 1))
        d_steps.extend([i for i in np.arange(-MAX_ROAD_WIDTH / 2, 0, d_step)])
        d_steps.extend([i for i in np.arange(d_step, MAX_ROAD_WIDTH / 2 + 0.001, d_step)])

    active_paths = []
    for di in d_steps:
        Ti = MAXT

        fp = Frenet_path()

        lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

        fp.t = [t for t in np.arange(0.0, Ti + 0.0001, DT)]  # add a small amount to Ti to ensure we get the last value
        fp.d = [lat_qp.calc_point(t) for t in fp.t]
        fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
        fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
        fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

        lon_qp = quartic_polynomial(s0, c_speed, a, TARGET_SPEED, 0.0, Ti)

        fp.s = [lon_qp.calc_point(t) for t in fp.t]
        fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
        fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
        fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

        Jp = sum(np.power(fp.d_ddd, 2))  # square of jerk
        Js = sum(np.power(fp.s_ddd, 2))  # square of jerk

        # square of diff from target speed
        ds = (TARGET_SPEED - fp.s_d[-1]) ** 2

        fp.cd = KJ * Jp + KT * MAXT + KD * fp.d[-1] ** 2
        fp.cv = KJ * Js + KT * MAXT + KD * ds
        fp.cf = KLON * fp.cv + KLAT * fp.cd

        active_paths.append(fp)
    frenet_paths.append(active_paths)

    # finally construct a double length path for forecasting
    if planner_args.GENERATE_PLANNING_PATH:
        di = 0
        Ti = MAXT * 2
        fp = Frenet_path()
        lat_qp = quintic_polynomial(0, 0, 0, di, 0.0, 0.0, Ti)

        fp.t = [t for t in np.arange(0.0, Ti, DT)]
        fp.d = [lat_qp.calc_point(t) for t in fp.t]
        fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
        fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
        fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

        lon_qp = quartic_polynomial(s0, TARGET_SPEED, 0.0, TARGET_SPEED, 0.0, Ti)

        fp.s = [lon_qp.calc_point(t) for t in fp.t]
        fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
        fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
        fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

        Jp = sum(np.power(fp.d_ddd, 2))  # square of jerk
        Js = sum(np.power(fp.s_ddd, 2))  # square of jerk

        # square of diff from target speed
        ds = (TARGET_SPEED - fp.s_d[-1]) ** 2

        fp.cd = KJ * Jp + KT * Ti + KD * fp.d[-1] ** 2
        fp.cv = KJ * Js + KT * Ti + KD * ds
        fp.cf = KLON * fp.cv + KLAT * fp.cd

        frenet_paths.append([fp])

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fps in fplist:
        for fp in fps:
            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = csp.calc_position(fp.s[i])
                if ix is None:
                    # ran off the end of the path -- fill in the rest of the path with the last point
                    fp.x.extend([fp.x[-1]] * (len(fp.s) - i))
                    fp.y.extend([fp.y[-1]] * (len(fp.s) - i))
                    break

                iyaw = csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(iyaw + math.pi / 2.0)
                fy = iy + di * math.sin(iyaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)

            # calc yaw and ds
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.sqrt(dx**2 + dy**2))

            try:
                fp.yaw.append(fp.yaw[-1])
                fp.ds.append(fp.ds[-1])
            except IndexError:
                pass  # empty trajectory

            # calc curvature
            for i in range(len(fp.yaw) - 1):
                try:
                    fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
                except ZeroDivisionError:
                    fp.c.append(0)

    return fplist


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2) for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS**2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob, args):
    MAX_SPEED = args.MAX_SPEED
    MAX_ACCEL = args.MAX_ACCEL
    MAX_CURVATURE = args.MAX_CURVATURE

    okind = []
    for i in range(len(fplist)):
        # if fplist[i].stopping:
        #     if any([v > 2*MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
        #         continue
        #     okind.append(i)
        #     continue
        # if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
        #     continue
        # elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
        #     continue
        # elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
        #     continue
        # elif not check_collision(fplist[i], ob):
        #     continue

        okind.append(i)

    return [fplist[i] for i in okind]


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, a, args=PlannerArgs()):
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, a, s0, args)
    fplist = calc_global_paths(fplist, csp)
    # fplist = check_paths(fplist, ob, args)

    # # find minimum cost path
    # mincost = float("inf")
    # bestpath = None
    # for fp in fplist:
    #     if mincost >= fp.cf:
    #         mincost = fp.cf
    #         bestpath = fp

    return fplist


def generate_target_course(points):
    csp = cubic_spline_planner.Spline2D(x=points[:, 0], y=points[:, 1])
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))
    return rx, ry, ryaw, rk, csp
