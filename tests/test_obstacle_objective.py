import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from obstacles.circular_obstacle_map import CircularObstacleMap
from objectives.obstacle_avoidance import ObstacleAvoidance
from trajectory.trajectory import Trajectory
from dotmap import DotMap


def create_params():
    p = DotMap()
    # Obstacle avoidance parameters
    p.avoid_obstacle_objective = DotMap(obstacle_margin=0.3,
                                        power=2,
                                        obstacle_cost=25.0)
    return p


def test_avoid_obstacle():
    # Create parameters
    p = create_params()
    
    # Create a circular obstacle map
    map_bounds = [(-2., -2.), (2., 2.)]  # [(min_x, min_y), (max_x, max_y)]
    cs = np.array([[-.5, -.5], [0.5, 0.5]])
    rs = np.array([[.5], [.5]])
    obstacle_map = CircularObstacleMap(map_bounds, cs, rs)
    
    # Define the objective
    objective = ObstacleAvoidance(params=p.avoid_obstacle_objective,
                                  obstacle_map=obstacle_map)
    
    # Define a set of positions and evaluate objective
    pos_nk2 = tf.constant([[[-0.5, -0.5], [0., 0.], [-0.5, 0.5]]], dtype=tf.float32)
    trajectory = Trajectory(dt=0.1, k=3, position_nk2=pos_nk2)
    
    # Compute the objective
    objective_values_13 = objective.evaluate_objective(trajectory)
    assert objective_values_13.shape == (1, 3)
    
    # Expected objective values
    expected_infringement = np.array([0.8, 0.8 - 0.5*np.sqrt(2), 0.])
    expected_objective = 25. * expected_infringement * expected_infringement

    assert np.allclose(objective_values_13.numpy()[0], expected_objective, atol=1e-4)


if __name__ == '__main__':
    test_avoid_obstacle()
