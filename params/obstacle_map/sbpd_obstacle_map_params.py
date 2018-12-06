from dotmap import DotMap
from obstacles.sbpd_map import SBPDMap
from params.renderer_params import create_params as create_renderer_params


def create_params():
    p = DotMap()

    # Load the dependencies
    p.renderer_params = create_renderer_params()

    p.obstacle_map = SBPDMap

    # Origin is always 0,0 for SBPD
    p.map_origin_2 = [0, 0]

    # Threshold distance from the obstacles to sample the start and the goal positions.
    p.sampling_thres = 2

    # Number of grid steps around the start position to use for plotting
    p.plotting_grid_steps = 100
    return p
