import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import pdb

def test_circular_occupancy_grid():
  from grids.circular_occupancy_grid import CircularOccupancyGrid
  np.random.seed(seed=11)
  l = 3#lxl meter grid
  min_r, max_r = .25, .5 #circles with radius between min_r and max_r
  grid = CircularOccupancyGrid(l, n=4, min_r=min_r, max_r=max_r)
  min_dist = grid.dist_to_nearest_obs((0., 0.))

  fig = plt.figure()
  ax = fig.add_subplot(111)
  grid.render(ax)
  plt.show()
 
if __name__ == '__main__':
  test_circular_occupancy_grid()
