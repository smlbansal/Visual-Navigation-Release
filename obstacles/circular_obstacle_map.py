from obstacles.obstacle_map import ObstacleMap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class CircularObstacleMap(ObstacleMap):
  def __init__(self, map_bounds, min_n, max_n, min_r, max_r):
    assert(min_r > 0 and max_r > 0)
    x_min, y_min, x_max, y_max = map_bounds[0][0], map_bounds[0][1], map_bounds[1][0], map_bounds[1][1]
    
    self.m = np.random.randint(min_n, max_n+1)
    self.map_bounds = map_bounds
   
    cs,rs = [],[] 
    while len(rs) < self.m:
      x, y = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)
      r = np.random.uniform(min_r, max_r)
      if np.sqrt(x**2+y**2) > r: #check obstacle doesn't touch origin
        cs.append([x,y])
        rs.append([r])
   
    cs, rs = np.array(cs), np.array(rs)
    self.c_m2= tf.convert_to_tensor(cs, name='circle_centers', dtype=tf.float32)
    self.r_m1 = tf.convert_to_tensor(rs, name='circle_radii', dtype=tf.float32) 
  
  def dist_to_nearest_obs(self, traj_nk2):
    with tf.name_scope('dist_to_obs'):
      c_11m2 = tf.expand_dims(tf.expand_dims(self.c_m2, axis=0), axis=0)
      r_11m1 = tf.expand_dims(tf.expand_dims(self.r_m1, axis=0), axis=0)
      traj_nk12 = tf.expand_dims(traj_nk2, axis=2) 
      
      c_nkm2 = c_11m2 + 0.*traj_nk12
      r_nkm = tf.squeeze(r_11m1 + 0.*c_nkm2[:,:,:,0:1])
      traj_nkm2 = traj_nk12 + 0.*c_nkm2
      
      diff_nkm2 = tf.square(c_nkm2 - traj_nkm2)
      dist_nkm = tf.sqrt(tf.reduce_sum(diff_nkm2, axis=3))-r_nkm
      min_dist_nk = tf.reduce_min(dist_nkm, axis=2) 
      return min_dist_nk

  def create_occupancy_grid(self):
    raise NotImplementedError

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
