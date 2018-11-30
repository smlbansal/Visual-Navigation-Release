from dotmap import DotMap
from utils import utils
import numpy as np
from waypoint_grids.uniform_sampling_grid import UniformSamplingGrid

dependencies = []


def load_params():
    # Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency) for dependency in dependencies})

    p.grid = UniformSamplingGrid

    # Parameters for the uniform sampling grid
    # Desired number of waypoints. Actual number may differ slightly
    # See ./waypoint_grids/uniform_sampling_grid.py for more info
    p.num_waypoints = 20000
    p.num_theta_bins = 21
    p.bound_min = [0., -2.5, -np.pi / 2]
    p.bound_max = [2.5, 2.5, np.pi / 2]
    
    # Additional parameters for the projected grid from the image space to the world coordinates
    p.projected_grid_params = DotMap(
                                    # Focal length in meters
                                    f=1,
                                    
                                    # Half-field of view
                                    fov=np.pi/6,
        
                                    # Height of the camera from the ground in meters
                                    h=1,
    )
    
    return p
