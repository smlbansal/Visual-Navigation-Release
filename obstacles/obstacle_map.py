import numpy as np


class ObstacleMap(object):
    name = 'ObstacleMapBase'

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

    def _point_to_map(self, pos_2, cast_to_int=False):
        """
        Convert pos_2 in real world coordinates
        to a point on the map.
        """
        map_pos_2 = pos_2/self.p.dx - self.p.map_origin_2
        if cast_to_int:
            map_pos_2 = map_pos_2.astype(np.int32)
        return map_pos_2

    def _map_to_point(self, pos_2, dtype=np.float32):
        """
        Convert pos_2 in map coordinates
        to a real world coordinate.
        """
        world_pos_2 = (pos_2 + self.p.map_origin_2)*self.p.dx
        return world_pos_2.astype(dtype)

