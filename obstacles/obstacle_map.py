import numpy as np


class ObstacleMap(object):

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        mb = p.map_bounds
        dx = p.dx
        origin_x = int(mb[0][0] / dx)
        origin_y = int(mb[0][1] / dx)
        p.map_origin_2 = np.array(
            [origin_x, origin_y], dtype=np.int32)

        Nx = int((mb[1][0] - mb[0][0]) / dx)
        Ny = int((mb[1][1] - mb[0][1]) / dx)
        p.map_size_2 = [Nx, Ny]

        return p

    def dist_to_nearest_obs(self, pos_nk2):
        raise NotImplementedError

    def create_occupancy_grid(self, xs_nn, ys_nn):
        raise NotImplementedError
