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

    def create_occupancy_grid_for_map(self, xs_nn, ys_nn):
        """
        Return an occupancy grid for the entire obstacle map where
        1 corresponds to occupied space and 0 corresponds to
        free space.
        """
        raise NotImplementedError

    @staticmethod
    def create_occupancy_grid(pos_n3, **kwargs):
        """
        Create egocentric occupancy grids at the positions
        in pos_n3.
        """
        raise NotImplementedError

