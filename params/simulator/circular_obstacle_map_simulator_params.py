from simulators.circular_obstacle_map_simulator import CircularObstacleMapSimulator
from params.obstacle_map.circular_obstacle_map_params import create_params as create_obstacle_map_params
from params.simulator.simulator_params import create_params as create_simulator_params 


def create_params():
    p = create_simulator_params()

    # Load the dependencies
    p.obstacle_map_params = create_obstacle_map_params()

    p.simulator = CircularObstacleMapSimulator
    return p
