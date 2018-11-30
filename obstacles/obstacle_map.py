import numpy as np


class ObstacleMap(object):

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        return p
        
    def dist_to_nearest_obs(self, pos_nk2):
        raise NotImplementedError

    def create_occupancy_grid(self, xs_nn, ys_nn):
        raise NotImplementedError
