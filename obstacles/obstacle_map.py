class ObstacleMap(object):

    def dist_to_nearest_obs(self, pos_nk2):
        raise NotImplementedError

    def create_occupancy_grid(self, xs_nn, ys_nn):
        raise NotImplementedError
