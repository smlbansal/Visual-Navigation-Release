from dotmap import DotMap
from utils import utils
from obstacles.circular_obstacle_map import CircularObstacleMap

dependencies = []


def load_params():
    # Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency) for dependency in dependencies})

    p.obstacle_map = CircularObstacleMap

    # [[min_x, min_y], [max_x, max_y]]
    p.map_bounds = [[0.0, 0.0], [8.0, 8.0]]
    p.dx = .05  # grid discretization for FmmMap and Obstacle Occupancy Grid
    return p
