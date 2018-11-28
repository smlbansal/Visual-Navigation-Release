from dotmap import DotMap
from utils import utils
from obstacles.sbpd_map import SBPDMap

dependencies = ['camera_params']


def load_params():
    # Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency) for dependency in dependencies})

    p.obstacle_map = SBPDMap

    p.image_renderer = DotMap(dataset_name='sbpd',
    						  building_name='area3',
    						  flip=False)

    p.dx = .05  # grid discretization for FmmMap and Obstacle Occupancy Grid
    return p
