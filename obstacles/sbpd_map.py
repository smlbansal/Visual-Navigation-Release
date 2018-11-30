from obstacles.obstacle_map import ObstacleMap
import numpy as np
import tensorflow as tf
from sbpd.sbpd_renderer import SBPDRenderer
from utils.fmm_map import FmmMap


class SBPDMap(ObstacleMap):
    def __init__(self, params):
        """
        Initialize a map for Stanford Building Parser Dataset (SBPD)
        """
        self.p = params
        self._r = SBPDRenderer(self.p.renderer_params)
        self._initialize_occupancy_grid()
        self._initialize_fmm_map()

    def _initialize_occupancy_grid(self):
        """
        Initialize the occupancy grid and associated parameters/
        instance variables
        """
        resolution, traversible = self._r.get_config()

        self.p.dx = resolution / 100.  # To convert to metres.

        # Reverse the shape as indexing into the traverible is reversed ([y, x] indexing)
        self.p.map_size_2 = np.array(traversible.shape[::-1])

        # [[min_x, min_y], [max_x, max_y]]
        self.map_bounds = [[0., 0.],  self.p.map_size_2*self.p.dx]

        free_xy = np.array(np.where(traversible)).T
        self.free_xy_m2 = free_xy[:, ::-1]

        self.occupancy_grid = np.logical_not(traversible)*1.

    def _initialize_fmm_map(self):
        """
        Initialize an FMM Map where 0 level set encodes the obstacle
        positions.
        """
        p = self.p
        occupied_xy_m2 = np.array(np.where(self.occupancy_grid)).T
        occupied_xy_m2 = occupied_xy_m2[:, ::-1]
        occupied_xy_m2_world = self._map_to_point(occupied_xy_m2)
        self.fmm_map = FmmMap.create_fmm_map_based_on_goal_position(
                                goal_positions_n2=occupied_xy_m2_world,
                                map_size_2=p.map_size_2,
                                dx=p.dx,
                                map_origin_2=p.map_origin_2,
                                mask_grid_mn=None)

    def dist_to_nearest_obs(self, pos_nk2):
        with tf.name_scope('dist_to_obs'):
            distance_nk = self.fmm_map.fmm_distance_map.compute_voxel_function(pos_nk2)
            return distance_nk

    def sample_point_112(self, rng):
        """
        Samples a real world x, y point in free space on the map.
        """
        idx = rng.choice(len(self.free_xy_m2))
        pos_112 = self.free_xy_m2[idx][None, None]
        return self._map_to_point(pos_112)

    def create_occupancy_grid(self):
        """
        Return the occupancy grid for the SBPD map.
        """
        return self.occupancy_grid

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

    def get_observation(self, config):
        """
        Render the robot's observation from system configuration config.
        """
        pos_map_n12 = self._point_to_map(config.position_nk2())
        heading_n11 = config.heading_nk1()

        starts_n2 = pos_map_n12[:, 0].numpy()
        thetas_n1 = heading_n11[:, 0].numpy()

        imgs = self._r.render_images(starts_n2, thetas_n1)
        return imgs

    def render(self, ax, start_config=None):
        p = self.p
        ax.imshow(self.occupancy_grid, cmap='gray_r',
                  extent=np.array(self.map_bounds).flatten(order='F'),
                  vmax=1.5, vmin=-.5, origin='lower')

        if start_config is not None:
            start_2 = start_config.position_nk2()[0, 0].numpy()
            delta = p.plotting_grid_steps * p.dx
            ax.set_xlim(start_2[0]-delta, start_2[0]+delta)
            ax.set_ylim(start_2[1]-delta, start_2[1]+delta)
