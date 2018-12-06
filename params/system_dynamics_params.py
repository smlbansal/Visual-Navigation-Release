from utils import utils
from dotmap import DotMap
from systems.dubins_v2 import DubinsV2


def create_params():
    p = DotMap()
    p.system = DubinsV2
    p.dt = .05
    p.v_bounds = [0.0, .6]
    p.w_bounds = [-1.1, 1.1]

    return p
