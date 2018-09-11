import numpy as np

class ObstacleGrid:

  def dist_to_nearest_obs(self, state):
    raise NotImplementedError

  def create_occupancy_grid(self):
    raise NotImplementedError
