from obstacles.obstacle_map import ObstacleMap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from systems.dubins_car import DubinsCar


class CircularObstacleMap(ObstacleMap):
    def __init__(self, map_bounds, centers_m2, radii_m1, params):
        """ initialize a circular obstacle grid
        with map bounds [(x_min,y_min), (x_max,y_max)]
        circle centers centers_m2=[[x_1,y_1],...,[x_2,y_2]] and radii radii_m1=[[r1],[r2],...]
        """
        self.map_bounds = map_bounds
        self.num_obstacles = len(radii_m1)
        self.obstacle_centers_m2 = tf.constant(centers_m2, name='circle_centers', dtype=tf.float32)
        self.obstacle_radii_m1 = tf.constant(radii_m1, name='circle_radii', dtype=tf.float32)

        self.p = self.parse_params(params)

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

    @classmethod
    def init_random_map(cls, map_bounds, rng, reset_params, params):
        min_n, max_n, min_r, max_r = reset_params.min_n, reset_params.max_n, reset_params.min_r, reset_params.max_r
        assert(min_r > 0 and max_r > 0)
        num_obstacles = rng.randint(min_n, max_n+1)
        return cls(map_bounds=map_bounds,
                   centers_m2=rng.uniform(map_bounds[0], map_bounds[1], (num_obstacles, 2)),
                   radii_m1=rng.uniform(min_r, max_r, (num_obstacles, 1)),
                   params=params)

    def dist_to_nearest_obs(self, pos_nk2):
        with tf.name_scope('dist_to_obs'):
            obstacle_centers_11m2 = self.obstacle_centers_m2[tf.newaxis, tf.newaxis, :, :]
            pos_nk12 = pos_nk2[:, :, tf.newaxis, :]
            distance_to_centers_nkm2 = tf.norm(pos_nk12 - obstacle_centers_11m2, axis=3) - self.obstacle_radii_m1[:, 0]
            return tf.reduce_min(distance_to_centers_nkm2, axis=2)

    def sample_point_112(self, rng):
        "Samples an x, y point on the map"""
        mb = self.map_bounds
        goal_x = rng.uniform(mb[0][0], mb[1][0])
        goal_y = rng.uniform(mb[0][1], mb[1][1])
        return np.array([goal_x, goal_y], dtype=np.float32)[None, None]

    def create_occupancy_grid_for_map(self):
        """ Creates an occupancy grid for the entire circular obstacle map
        where 0 and 1 represent free and occupied space respectively.
        """
        mb = self.p.map_bounds
        Nx, Ny = self.p.map_size_2
        xx, yy = np.meshgrid(np.linspace(mb[0][0], mb[1][0], Nx),
                             np.linspace(mb[0][1], mb[1][1], Ny),
                             indexing='xy')

        xs_nm = tf.constant(xx, dtype=tf.float32)
        ys_nm = tf.constant(yy, dtype=tf.float32)

        grid_nm2 = tf.stack([xs_nm, ys_nm], axis=2)
        dists_nm = self.dist_to_nearest_obs(grid_nm2)
        occupancy_grid_nm = tf.nn.relu(tf.sign(dists_nm*-1))
        return occupancy_grid_nm

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Render the robot's observation (occupancy_grid) from system configuration config
        or pos_nk3. If obs_centers_nl2 and obs_radii_nl1 are passed as arguments
        renders the occupancy_grid for these parameters, otherwise renders based
        on the current state of the instance variables.
        """
        # One of config and pos_nk3 must be not None
        assert((config is None) != (pos_n3 is None))

        if config is not None:
            pos_n3 = config.position_and_heading_nk3()[:, 0].numpy()

        occupancy_grid_positions_ego_1mk12 = kwargs['occupancy_grid_positions_ego_1mk12']
        if 'obs_centers_nl2' in kwargs.keys():
            obs_centers_nl2 = kwargs['obs_centers_nl2']
            obs_radii_nl1 = kwargs['obs_radii_nl1']
        else:
            obs_centers_nl2 = self.obstacle_centers_m2[None]
            obs_radii_nl1 = self.obstacle_radii_m1[None]

        # Convert the obstacle centers to the egocentric coordinates
        # (here, we leverage the fact that circles after
        # axis rotation remain circles).
        n, l = obs_radii_nl1.shape[0], obs_radii_nl1.shape[1]
        obs_centers_ego_nl2 = DubinsCar.convert_position_and_heading_to_ego_coordinates(
            pos_n3[:, np.newaxis, :],
            np.concatenate([obs_centers_nl2, np.zeros((n, l, 1), dtype=np.float32)], axis=2))[:, :, :2]

        # Compute distance to the obstacles
        distance_to_centers_nmkl = tf.norm(obs_centers_ego_nl2[:, tf.newaxis, tf.newaxis, :, :] -
                                           occupancy_grid_positions_ego_1mk12, axis=4) \
                                   - obs_radii_nl1[:, tf.newaxis, tf.newaxis, :, 0]
        distance_to_nearest_obstacle_nmk1 = tf.reduce_min(distance_to_centers_nmkl, axis=3, keep_dims=True)
        occupancy_grid_nmk1 = 0.5 * (1. - tf.sign(distance_to_nearest_obstacle_nmk1))
        return occupancy_grid_nmk1.numpy()

    def render(self, ax):
        for i in range(self.num_obstacles):
            c = self.obstacle_centers_m2[i].numpy()
            r = self.obstacle_radii_m1[i][0].numpy()
            c = plt.Circle((c[0], c[1]), r, color='b')
            ax.add_artist(c)
        map_bounds = self.map_bounds
        x_min, y_min, x_max, y_max = map_bounds[0][0], map_bounds[0][1], map_bounds[1][0], map_bounds[1][1]
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_title('Obstacle Map')

    def render_with_obstacle_margins(self, ax, margin0=.3, margin1=.5):
        """ Render the map with different opacity circles indicating the intensity of the cost
        
        function around obstacles"""
        for i in range(self.num_obstacles):
            c = self.obstacle_centers_m2[i].numpy()
            r = self.obstacle_radii_m1[i][0].numpy()
            c_actual = plt.Circle((c[0], c[1]), r, color='b')
            c0 = plt.Circle((c[0], c[1]), r+margin0, color='b')
            c1 = plt.Circle((c[0], c[1]), r+margin1, color='b')

            c0.set_alpha(.4)
            c1.set_alpha(.2)
            ax.add_artist(c_actual)
            ax.add_artist(c0)
            ax.add_artist(c1)

        map_bounds = self.map_bounds
        x_min, y_min, x_max, y_max = map_bounds[0][0], map_bounds[0][1], map_bounds[1][0], map_bounds[1][1]
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_title('Obstacle Map')
