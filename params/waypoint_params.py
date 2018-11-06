from dotmap import DotMap
from utils import utils
import numpy as np
from waypoint_grids.uniform_sampling_grid import UniformSamplingGrid

dependencies = []


def load_params():
    # Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency) for dependency in dependencies})

    p.grid = UniformSamplingGrid

    # Desired number of waypoints. Actual number may differ slightly
    # See ./waypoint_grids/uniform_sampling_grid.py for more info
    p.num_waypoints = 20000

    p.num_theta_bins = 21
    p.bound_min = [0., -2.5, -np.pi / 2]
    p.bound_max = [2.5, 2.5, np.pi / 2]
    return p


def parse_params(p):
    # Update the number of waypoints based on how many will actually be sampled
    p.n = p.grid.compute_number_waypoints(p)
    return p
