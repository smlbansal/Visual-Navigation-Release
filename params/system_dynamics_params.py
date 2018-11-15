from utils import utils
from dotmap import DotMap
from systems.dubins_v2 import DubinsV2

dependencies = []


def load_params():
    # Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency)
                for dependency in dependencies})

    p.system = DubinsV2
    p.dt = .05
    p.v_bounds = [0.0, .6]
    p.w_bounds = [-1.1, 1.1]

    return p
