from numpy import pi as PI

# --------------------------------------------------------------------------------
# DEBUG flags
# --------------------------------------------------------------------------------
DEBUG_INFORMATION_GAIN = True
DEBUG_VISIBILITY = True
DEBUG_TRAJECTORIES = True

# --------------------------------------------------------------------------------
# Figure Constants
# --------------------------------------------------------------------------------
FIG_LIDAR_MAP = 1
FIG_OCCUPANCY_GRID = 2
FIG_DYNAMIC_OCCUPANCY_GRID = 3
FIG_TRAJECTORIES = 4
FIG_VISIBILITY = 5
FIG_VISIBILITY_COSTMAP = 6


# --------------------------------------------------------------------------------
# Path to the directory with all components
# --------------------------------------------------------------------------------
DEFAULT_CACHE_LOC = "./cache"
CACHE_PATH = "v1/perception/1_4_3/training"
MAP_PATH = "maps"

# --------------------------------------------------------------------------------
# LIDAR Constants
# --------------------------------------------------------------------------------
LIDAR_RANGE = 75.0
LIDAR_RAYS = 2650.0
LIDAR_INCREMENT = (PI * 2.0) / LIDAR_RAYS

# --------------------------------------------------------------------------------
# Grid Constants
# --------------------------------------------------------------------------------
GRID_WIDTH = 2 * LIDAR_RANGE
GRID_CELL_WIDTH = 0.5
GRID_SIZE = int(GRID_WIDTH / GRID_CELL_WIDTH)

LIDAR_LOWER_X_BOUND = -GRID_WIDTH / 2
LIDAR_LOWER_Y_BOUND = -GRID_WIDTH / 2
LIDAR_LOWER_Z_BOUND = 0.0
LIDAR_UPPER_X_BOUND = GRID_WIDTH / 2
LIDAR_UPPER_Y_BOUND = GRID_WIDTH / 2
LIDAR_UPPER_Z_BOUND = 5.0

Z_MIN = 0.5
Z_MAX = 2.0

# the voxel map is scaled at 0.2 m/pixel -- use that value to scale the
# voxel cube
VOXEL_MAP_SCALE = 0.1
VOXEL_MAP_X = int((LIDAR_UPPER_X_BOUND - LIDAR_LOWER_X_BOUND) / VOXEL_MAP_SCALE)
VOXEL_MAP_Y = int((LIDAR_UPPER_Y_BOUND - LIDAR_LOWER_Y_BOUND) / VOXEL_MAP_SCALE)
VOXEL_MAP_Z = int((LIDAR_UPPER_Z_BOUND - LIDAR_LOWER_Z_BOUND) / VOXEL_MAP_SCALE)

MAX_PEDESTRIAN_SPEED = 1.5

OCCUPANCY_THRESHOLD = 0.6
RISK_THRESHOLD = 0.1

LANE_WIDTH = 4.0
