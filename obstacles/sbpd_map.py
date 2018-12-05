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
        self._r = SBPDRenderer.get_renderer(self.p.renderer_params)
        self._initialize_occupancy_grid_for_map()
        self._initialize_fmm_map()

    def _initialize_occupancy_grid_for_map(self):
        """
        Initialize the occupancy grid for the entire map and
        associated parameters/ instance variables
        """
        resolution, traversible = self._r.get_config()

        self.p.dx = resolution / 100.  # To convert to metres.

        # Reverse the shape as indexing into the traverible is reversed ([y, x] indexing)
        self.p.map_size_2 = np.array(traversible.shape[::-1])

        # [[min_x, min_y], [max_x, max_y]]
        self.map_bounds = np.array([[0., 0.],  self.p.map_size_2*self.p.dx])

        free_xy = np.array(np.where(traversible)).T
        self.free_xy_map_m2 = free_xy[:, ::-1]

        self.occupancy_grid_map = np.logical_not(traversible)*1.

    def _initialize_fmm_map(self):
        """
        Initialize an FMM Map where 0 level set encodes the obstacle
        positions.
        """
        p = self.p
        occupied_xy_m2 = np.array(np.where(self.occupancy_grid_map)).T
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

    def sample_point_112(self, rng, free_xy_map_m2=None):
        """
        Samples a real world x, y point in free space on the map.
        Optionally the user can pass in free_xy_m2 a list of m (x, y)
        points from which to sample.
        """
        if free_xy_map_m2 is None:
            free_xy_map_m2 = self.free_xy_map_m2

        idx = rng.choice(len(free_xy_map_m2))
        pos_112 = free_xy_map_m2[idx][None, None]
        return self._map_to_point(pos_112)

    def create_occupancy_grid_for_map(self, xs_nn=None, ys_nn=None):
        """
        Return the occupancy grid for the SBPD map.
        """
        return self.occupancy_grid_map

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Render the robot's observation from system configuration config
        or pos_nk3.
        """
        # One of config and pos_nk3 must be not None
        assert((config is None) != (pos_n3 is None))

        if config is not None:
            pos_n3 = config.position_and_heading_nk3()[:, 0].numpy()

        starts_n2 = self._point_to_map(pos_n3[:, :2])
        thetas_n1 = pos_n3[:, 2:3]

        imgs = self._r.render_images(starts_n2, thetas_n1, **kwargs)
        return imgs

    def render(self, ax, start_config=None):
        p = self.p
        ax.imshow(self.occupancy_grid_map, cmap='gray_r',
                  extent=np.array(self.map_bounds).flatten(order='F'),
                  vmax=1.5, vmin=-.5, origin='lower')

        if start_config is not None:
            start_2 = start_config.position_nk2()[0, 0].numpy()
            delta = p.plotting_grid_steps * p.dx
            ax.set_xlim(start_2[0]-delta, start_2[0]+delta)
            ax.set_ylim(start_2[1]-delta, start_2[1]+delta)

    def render_with_obstacle_margins(self, ax, start_config=None, margin0=.3, margin1=.5):
        p = self.p
        occupancy_grid_masked = np.ma.masked_where(self.occupancy_grid_map == 0,
                                                   self.occupancy_grid_map)
        ax.imshow(occupancy_grid_masked, cmap='Blues_r',
                  extent=np.array(self.map_bounds).flatten(order='F'),
                  origin='lower', vmax=2.0)

        self._render_margin(ax, margin=margin0, alpha=.5)
        self._render_margin(ax, margin=margin1, alpha=.35)

        if start_config is not None:
            start_2 = start_config.position_nk2()[0, 0].numpy()
            delta = p.plotting_grid_steps * p.dx
            ax.set_xlim(start_2[0]-delta, start_2[0]+delta)
            ax.set_ylim(start_2[1]-delta, start_2[1]+delta)

    def _render_margin(self, ax, margin, alpha):
        """
        Render a margin around the occupied space indicating the intensity
        of the obstacle avoidance cost function.
        """
        xs = np.arange(self.map_bounds[0][0], self.map_bounds[1][0], self.p.dx)
        ys = np.arange(self.map_bounds[0][1], self.map_bounds[1][1], self.p.dx)
        xs, ys = np.meshgrid(xs, ys)
        xs = xs.ravel()
        ys = ys.ravel()
        pos_n12 = np.stack([xs, ys], axis=1)[:, None]
        dists_nk = self.dist_to_nearest_obs(pos_n12).numpy()

        margin_mask_n = (dists_nk < margin)[:, 0]
        margin_mask_mn = margin_mask_n.reshape(self.occupancy_grid_map.shape)
        mask = np.logical_and(self.occupancy_grid_map, margin_mask_mn == 0)

        margin_img = np.ma.masked_where(mask, margin_mask_mn)
        ax.imshow(margin_img, cmap='Blues',
                  extent=np.array(self.map_bounds).flatten(order='F'),
                  origin='lower', alpha=alpha, vmax=2.0)
