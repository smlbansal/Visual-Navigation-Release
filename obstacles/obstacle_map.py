class ObstacleMap:

  def dist_to_nearest_obs(self, traj):
    raise NotImplementedError

  def create_occupancy_grid(self):
    raise NotImplementedError
