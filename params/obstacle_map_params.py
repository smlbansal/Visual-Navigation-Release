from dotmap import DotMap
from utils import utils
from obstacles.sbpd_map import SBPDMap

dependencies = ['renderer_params']


def load_params():
    # Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency) for dependency in dependencies})

    p.obstacle_map = SBPDMap

    # Origin is always 0,0 for SBPD
    p.map_origin_2 = [0, 0]

    # Threshold distance from the obstacles to sample the start and the goal positions.
    p.sampling_thres = 2

    # Number of grid steps around the start position to use for plotting
    p.plotting_grid_steps = 100
    return p
