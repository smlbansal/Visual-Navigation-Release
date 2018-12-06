from dotmap import DotMap
from simulators.sbpd_simulator import SBPDSimulator
from params.obstacle_map.sbpd_obstacle_map_params import create_params as create_obstacle_map_params
from params.simulator.simulator_params import create_params as create_simulator_params 


def create_params():
    p = create_simulator_params()

    # Load the dependencies
    p.obstacle_map_params = create_obstacle_map_params()

    p.simulator = SBPDSimulator

    # Custom goal reset parameters
    # 'random_v1 ': the goal position is initialized randomly on the
    # map but at least at a distance of the obstacle margin from the
    # obstacle and at most max_dist from the start. Additionally
    # the difference between fmm and l2 distance between goal and
    # start must be greater than some threshold (sampled based on
    # max_dist_diff)
    p.reset_params.goal_config = DotMap(position=DotMap(
                                                    reset_type='random_v1',
                                                    max_dist_diff=.5,
                                                    max_fmm_dist=6.0
                                                ))
    return p
