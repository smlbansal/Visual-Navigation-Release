from dotmap import DotMap
from obstacles.circular_obstacle_map import CircularObstacleMap
from params.renderer_params import create_params as create_renderer_params


def create_params():
    p = DotMap()

    # Load the dependencies
    p.renderer_params = create_renderer_params()

    # The Circular Obstacle Map only supports occupancy grid as observation
    assert(p.renderer_params.camera_params.modalities[0] == 'occupancy_grid')

    p.obstacle_map = CircularObstacleMap

    # [[min_x, min_y], [max_x, max_y]]
    p.map_bounds = [[0.0, 0.0], [8.0, 8.0]]
    p.dx = .05  # grid discretization for FmmMap and Obstacle Occupancy Grid
    return p
