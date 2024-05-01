# ported from github:waterlooRobotics/mobileRobotics
#
# originally written for Matlab and ported to Python

import numpy as np
from math import floor


def bresenham(x1, y1, x2, y2):
    '''
    vectorized/optmized version of Bresenham line algorithm. No loops.
    Format:
                   (x,y) = bham(x1,y1,x2,y2)

    Input:
                   (x1,y1): Start position
                   (x2,y2): End position

    Output:
                   x y: the line coordinates from (x1,y1) to (x2,y2)

    Usage example:
                   [x y]=bham(1,1, 10,-5);
                   plot(x,y,'or');

    '''

    x1 = round(x1)
    x2 = round(x2)
    y1 = round(y1)
    y2 = round(y2)

    dx = abs(x2-x1)
    dy = abs(y2-y1)

    steep = dy > dx
    if steep:
        t = dx
        dx = dy
        dy = t

    if dy == 0:
        q = np.zeros([int(dx+1), 1])
    else:
        q = [a for a in np.arange(floor(dx/2), -dy*dx+floor(dx/2)-dy, -dy)]
        q = np.hstack([0, (np.diff(np.mod(q, dx)) >= 0).astype(float)])

    if steep:
        if y1 <= y2:
            y = np.arange(y1, y2+1).astype(int)
        else:
            y = np.arange(y1, y2-1, -1).astype(int)
        if x1 <= x2:
            x = x1+np.cumsum(q).astype(int)
        else:
            x = x1-np.cumsum(q).astype(int)
    else:
        if x1 <= x2:
            x = np.arange(x1, x2+1).astype(int)
        else:
            x = np.arange(x1, x2-1, -1).astype(int)
        if y1 <= y2:
            y = y1+np.cumsum(q).astype(int)
        else:
            y = y1-np.cumsum(q).astype(int)

    return x, y
