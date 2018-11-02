from dotmap import DotMap
from utils import utils
import numpy as np
from obstacles.circular_obstacle_map import CircularObstacleMap

dependencies = []

def load_params():
    # Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency) for dependency in dependencies})

    p.classname = CircularObstacleMap

    # [[min_x, min_y], [max_x, max_y]]
    p.map_bounds = [[0.0, 0.0], [8.0, 8.0]]
    p.dx = .05 # grid discretization for FmmMap and Obstacle Occupancy Grid
    return p

def parse_params(p):
    mb = p.map_bounds
    dx = p.dx
    origin_x = int(mb[0][0] / dx)
    origin_y = int(mb[0][1] / dx)
    p.map_origin_2 = np.array(
        [origin_x, origin_y], dtype=np.int32)

    Nx = int((mb[1][0] - mb[0][0]) / dx)
    Ny = int((mb[1][1] - mb[0][1]) / dx)
    p.map_size_2 = [Nx, Ny]
    return p