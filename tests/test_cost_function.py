import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from obstacles.circular_obstacle_map import CircularObstacleMap
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.goal_distance import GoalDistance
from objectives.angle_distance import AngleDistance
from objectives.objective_function import ObjectiveFunction
from trajectory.trajectory import Trajectory
from utils.fmm_map import FmmMap
from dotmap import DotMap


def create_params():
    p = DotMap()
    # Obstacle avoidance parameters
    p.avoid_obstacle_objective = DotMap(obstacle_margin=0.3,
                                        power=2,
                                        obstacle_cost=25.0)
    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=1,
                                    angle_cost=25.0)
    # Goal Distance parameters
    p.goal_distance_objective = DotMap(power=2,
                                       goal_cost=25.0)
    return p


def test_cost_function():
    # Create parameters
    p = create_params()
    
    # Create a circular obstacle map
    map_bounds = [(-2., -2.), (2., 2.)]  # [(min_x, min_y), (max_x, max_y)]
    cs = np.array([[-.5, -.5], [0.5, 0.5]])
    rs = np.array([[.5], [.5]])
    obstacle_map = CircularObstacleMap(map_bounds, cs, rs)

    # Create the occupancy grid
    xx, yy = np.meshgrid(np.linspace(-2., 2., 40), np.linspace(-2., 2., 40), indexing='xy')
    obstacle_occupancy_grid = obstacle_map.create_occupancy_grid(tf.constant(xx, dtype=tf.float32),
                                                                 tf.constant(yy, dtype=tf.float32))
    assert obstacle_occupancy_grid.shape == (40, 40)

    # Define a goal position and compute the corresponding fmm map
    fmm_map = FmmMap.create_fmm_map_based_on_goal_position(goal_positions_n2=np.array([[0., 0.]]),
                                                           map_size_2=np.array([40, 40]),
                                                           dx=0.1,
                                                           map_origin_2=np.array([-20., -20.]),
                                                           mask_grid_mn=obstacle_occupancy_grid)
    
    # Define the cost function
    objective_function = ObjectiveFunction()
    objective_function.add_objective(ObstacleAvoidance(params=p.avoid_obstacle_objective, obstacle_map=obstacle_map))
    objective_function.add_objective(GoalDistance(params=p.goal_distance_objective, fmm_map=fmm_map))
    objective_function.add_objective(AngleDistance(params=p.goal_angle_objective, fmm_map=fmm_map))
    
    # Define each objective separately
    objective1 = ObstacleAvoidance(params=p.avoid_obstacle_objective, obstacle_map=obstacle_map)
    objective2 = GoalDistance(params=p.goal_distance_objective, fmm_map=fmm_map)
    objective3 = AngleDistance(params=p.goal_angle_objective, fmm_map=fmm_map)
    
    # Define a set of positions and evaluate objective
    pos_nk2 = tf.constant([[[-1., 1.], [0.1, 0.1], [-0.1, -0.1]]], dtype=tf.float32)
    trajectory = Trajectory(dt=0.1, n=1, k=3, position_nk2=pos_nk2)
    
    # Compute the objective function
    values_by_objective = objective_function.evaluate_function_by_objective(trajectory)
    overall_objective = objective_function.evaluate_function(trajectory)
    
    # Expected objective values
    expected_objective1 = objective1.evaluate_objective(trajectory)
    expected_objective2 = objective2.evaluate_objective(trajectory)
    expected_objective3 = objective3.evaluate_objective(trajectory)
    expected_overall_objective = tf.reduce_mean(expected_objective1 + expected_objective2 + expected_objective3, axis=1)
    
    assert len(values_by_objective) == 3
    assert values_by_objective[0][1].shape == (1, 3)
    assert overall_objective.shape == (1,)
    assert np.allclose(overall_objective.numpy(), expected_overall_objective.numpy(), atol=1e-2)


if __name__ == '__main__':
    test_cost_function()
