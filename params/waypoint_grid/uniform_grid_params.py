from dotmap import DotMap
import numpy as np
from waypoint_grids.uniform_sampling_grid import UniformSamplingGrid


def create_params():
    p = DotMap()
    p.grid = UniformSamplingGrid

    # Parameters for the uniform sampling grid
    # Desired number of waypoints. Actual number may differ slightly
    # See ./waypoint_grids/uniform_sampling_grid.py for more info
    p.num_waypoints = 20000
    p.num_theta_bins = 21
    p.bound_min = [0., -2.5, -np.pi / 2]
    p.bound_max = [2.5, 2.5, np.pi / 2]
    
    return p
