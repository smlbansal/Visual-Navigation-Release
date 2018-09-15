from obstacles.obstacle_map import ObstacleMap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class CircularObstacleMap(ObstacleMap):
    def __init__(self, map_bounds, cs, rs):
        """ initialize a circular obstacle grid
        with map bounds [(x_min,y_min), (x_max,y_max)]
        circle centers cs=[[x_1,y_1],...,[x_2,y_2]] and radii rs=[[r1],[r2],...]
        """
        self.map_bounds, self.m = map_bounds, len(rs)
        self.c_m2= tf.convert_to_tensor(np.array(cs), name='circle_centers', dtype=tf.float32)
        self.r_m1 = tf.convert_to_tensor(np.array(rs), name='circle_radii', dtype=tf.float32) 
       
    @staticmethod 
    def init_random_map(map_bounds, min_n, max_n, min_r, max_r):
        assert(min_r > 0 and max_r > 0)
        x_min, y_min, x_max, y_max = map_bounds[0][0], map_bounds[0][1], map_bounds[1][0], map_bounds[1][1]

        m = np.random.randint(min_n, max_n+1)
        map_bounds = map_bounds

        cs,rs = [],[] 
        while len(rs) < m:
            x, y = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)
            r = np.random.uniform(min_r, max_r)
            if np.sqrt(x**2+y**2) > r: #check obstacle doesn't touch origin
                cs.append([x,y])
                rs.append([r])
        return CircularObstacleMap(map_bounds, cs, rs)

    def dist_to_nearest_obs(self, trajectory):
        pos_nk2 = trajectory.position_nk2()
        return self._dist_to_nearest_obs(pos_nk2)       
 
    def _dist_to_nearest_obs(self, pos_nk2):
        with tf.name_scope('dist_to_obs'):
            c_11m2 = tf.expand_dims(tf.expand_dims(self.c_m2, axis=0), axis=0)
            r_11m1 = tf.expand_dims(tf.expand_dims(self.r_m1, axis=0), axis=0)
            pos_nk12 = tf.expand_dims(pos_nk2, axis=2) 

            c_nkm2 = c_11m2 + 0.*pos_nk12
            r_nkm = tf.squeeze(r_11m1 + 0.*c_nkm2[:,:,:,0:1])
            pos_nkm2 = pos_nk12 + 0.*c_nkm2

            diff_nkm2 = tf.square(c_nkm2 - pos_nkm2)
            dist_nkm = tf.sqrt(tf.reduce_sum(diff_nkm2, axis=3))-r_nkm
            min_dist_nk = tf.reduce_min(dist_nkm, axis=2) 
            return min_dist_nk

    def create_occupancy_grid(self, xs_nn, ys_nn):
        """
            Creates an occupancy grid where 0 and 1 represent free
            and occupied space respectively
        """
        grid_nn2 = tf.stack([xs_nn, ys_nn], axis=2)
        dists_nn = self._dist_to_nearest_obs(grid_nn2)
        occupancy_grid_nn = tf.nn.relu(tf.sign(dists_nn*-1))
        return occupancy_grid_nn

    def render(self, ax):
        for i in range(self.m):
            c = self.c_m2[i].numpy()
            r = self.r_m1[i][0].numpy()
            c = plt.Circle((c[0], c[1]), r, color='b')
            ax.add_artist(c)
        ax.plot(0.0, 0.0, 'ro')
        map_bounds = self.map_bounds
        x_min, y_min, x_max, y_max = map_bounds[0][0], map_bounds[0][1], map_bounds[1][0], map_bounds[1][1]
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
