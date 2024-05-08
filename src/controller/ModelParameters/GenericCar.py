# Parameters for a generic car -- super simple
from numpy import pi, inf

LENGTH = 5.0
WIDTH = 2.0

MAX_V = 10
MIN_V = -10
MAX_W = pi
MIN_W = -pi

MIN_DELTA = -inf
MAX_DELTA = inf


class Vehicle:
    control_len = 2
    state_len = 3

    # set limits on velocity and turning
    min_v = MIN_V
    max_v = MAX_V
    min_w = MIN_W
    max_w = MAX_W
    max_delta = MAX_DELTA
    min_delta = MIN_DELTA

    def __init__(self) -> None:
        pass

    #   Step Functions
    #   --------------
    #        x(k+1) = x(k) + v cos(theta(k)),
    #        y(k+1) = y(k) + v sin(theta(k)),
    #        theta(k+1) = theta(k) + w,
    #  next_state = [v*cos(theta)*dt, v*sin(theta)*dt
    @staticmethod
    def ode(state, control):
        pass
