import tensorflow as tf

from objectives.objective_function import Objective


class ObstacleAvoidance(Objective):
    """
    Define the obstacle avoidance objective.
    """
    def __init__(self, params, obstacle_map):
        self.p = params
        self.obstacle_map = obstacle_map

    def evaluate_objective(self, trajectory):
        dist_to_obstacles_nk = self.obstacle_map.dist_to_nearest_obs(trajectory.position_nk3())
        infringement_nk = tf.nn.relu(self.p.obstacle_margin - dist_to_obstacles_nk)
        return self.p.obstacle_cost*tf.pow(infringement_nk, self.p.power)
