import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import pdb

def test_circular_obstacle_map():
  from obstacles.circular_obstacle_map import CircularObstacleMap
  np.random.seed(seed=1)
  n,k = 100, 20
  map_bounds = [(-2,-2), (2,2)] #[(min_x, min_y), (max_x, max_y)]
  min_n, max_n = 2, 4
  min_r, max_r = .25, .5

  traj_nk2 = np.zeros((n,k,2))
  traj_nk2 = tf.convert_to_tensor(traj_nk2, name='traj', dtype=tf.float32)

  grid = CircularObstacleMap(map_bounds, min_n, max_n, min_r, max_r)
  obs_dists_nk = grid.dist_to_nearest_obs(traj_nk2)
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  grid.render(ax)
  plt.show()
 
if __name__ == '__main__':
  test_circular_obstacle_map()
