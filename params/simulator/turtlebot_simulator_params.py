from dotmap import DotMap
from simulators.turtlebot_simulator import TurtlebotSimulator
from params.obstacle_map.turtlebot_obstacle_map_params import create_params as create_obstacle_map_params
from params.simulator.simulator_params import create_params as create_simulator_params


def create_params():
    p = create_simulator_params()

    # Load the dependencies
    p.obstacle_map_params = create_obstacle_map_params()

    p.simulator = TurtlebotSimulator

    p.reset_params.goal_config = DotMap(position=DotMap(
                                                reset_type='custom',
                                                goal_pos=[0.0, 0.0]
                                            ))
    return p
