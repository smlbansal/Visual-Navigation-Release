from dotmap import DotMap
from systems.turtlebot_dubins_v2 import TurtlebotDubinsV2


def create_params():
    p = DotMap()
    p.system = TurtlebotDubinsV2
    p.dt = .05
    p.v_bounds = [0.0, .6]
    p.w_bounds = [-1.1, 1.1]

    # Set the acceleration bounds such that
    # by default they are never hit
    p.linear_acc_max = 10e7
    p.angular_acc_max = 10e7

    p.simulation_params = DotMap(simulation_mode='realistic',
                                 noise_params = DotMap(is_noisy=False,
                                                       noise_type='uniform',
                                                       noise_lb=[-0.02, -0.02, 0.],
                                                       noise_ub=[0.02, 0.02, 0.],
                                                       noise_mean=[0., 0., 0.],
                                                       noise_std=[0.02, 0.02, 0.]))

    return p
