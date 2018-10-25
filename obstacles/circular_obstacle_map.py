from obstacles.obstacle_map import ObstacleMap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class CircularObstacleMap(ObstacleMap):
    def __init__(self, map_bounds, centers_m2, radii_m1):
        """ initialize a circular obstacle grid
        with map bounds [(x_min,y_min), (x_max,y_max)]
        circle centers centers_m2=[[x_1,y_1],...,[x_2,y_2]] and radii radii_m1=[[r1],[r2],...]
        """
        self.map_bounds = map_bounds
        self.num_obstacles = len(radii_m1)
        self.obstacle_centers_m2 = tf.constant(centers_m2, name='circle_centers', dtype=tf.float32)
        self.obstacle_radii_m1 = tf.constant(radii_m1, name='circle_radii', dtype=tf.float32)

    @classmethod
    def init_random_map(cls, map_bounds, min_n, max_n, min_r, max_r):
        assert(min_r > 0 and max_r > 0)
        num_obstacles = np.random.randint(min_n, max_n+1)
        return cls(map_bounds=map_bounds,
                   centers_m2=np.random.uniform(map_bounds[0], map_bounds[1], (num_obstacles, 2)),
                   radii_m1=np.random.uniform(min_r, max_r, (num_obstacles, 1)))

    def dist_to_nearest_obs(self, pos_nk2):
        with tf.name_scope('dist_to_obs'):
            obstacle_centers_11m2 = self.obstacle_centers_m2[tf.newaxis, tf.newaxis, :, :]
            pos_nk12 = pos_nk2[:, :, tf.newaxis, :]
            distance_to_centers_nkm2 = tf.norm(pos_nk12 - obstacle_centers_11m2, axis=3) - self.obstacle_radii_m1[:, 0]
            return tf.reduce_min(distance_to_centers_nkm2, axis=2)

    def sample_start_and_goal_12(self, rng, goal_radius=0.0, goal_norm=2, obs_margin=0.0):
        """ Samples a random start and goal point on the map
        of dimension (1,2). Ensures that |start-goal|_{goal_norm} > goal_radius"""
        start_112 = self._sample_point_112(rng)
        dist_to_obs = tf.squeeze(self.dist_to_nearest_obs(start_112))
        while dist_to_obs <= obs_margin:
            start_112 = self._sample_point_112(rng)
            dist_to_obs = tf.squeeze(self.dist_to_nearest_obs(start_112))

        goal_112 = self._sample_point_112(rng)
        dist_to_obs = tf.squeeze(self.dist_to_nearest_obs(goal_112))
        dist_to_goal = np.linalg.norm((start_112 - goal_112)[0], ord=goal_norm)
        while dist_to_obs <= obs_margin or dist_to_goal <= goal_radius:
            goal_112 = self._sample_point_112(rng)
            dist_to_obs = tf.squeeze(self.dist_to_nearest_obs(goal_112))
            dist_to_goal = np.linalg.norm((start_112 - goal_112)[0], ord=goal_norm)
        return start_112[0], goal_112[0]

    def _sample_point_112(self, rng):
        "Samples an x,y point on the map"""
        mb = self.map_bounds
        goal_x = rng.uniform(mb[0][0], mb[1][0])
        goal_y = rng.uniform(mb[0][1], mb[1][1])
        return np.array([goal_x, goal_y], dtype=np.float32)[None, None]

    def create_occupancy_grid(self, xs_nn, ys_nn):
        """
            Creates an occupancy grid where 0 and 1 represent free
            and occupied space respectively
        """
        grid_nn2 = tf.stack([xs_nn, ys_nn], axis=2)
        dists_nn = self.dist_to_nearest_obs(grid_nn2)
        occupancy_grid_nn = tf.nn.relu(tf.sign(dists_nn*-1))
        return occupancy_grid_nn

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
