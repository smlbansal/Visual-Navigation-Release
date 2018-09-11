from .obstacle import ObstacleGrid
import numpy as np
import matplotlib.pyplot as plt

class CircularOccupancyGrid(ObstacleGrid):
  def __init__(self, l, n, min_r, max_r):
    self.l = l
    self.n = n
    self.c, self.r = [], []
    i = 0
    while len(self.r) < n:
      x,y = np.random.uniform(-l, l, 2)
      r = np.random.uniform(min_r, max_r)
      if np.sqrt(x**2+y**2) > r:
        self.c.append(np.random.uniform(-l, l, 2))
        self.r.append(np.random.uniform(min_r, max_r))

  def dist_to_nearest_obs(self, state):
    dists = []
    for i in range(self.n):
      x,y = state
      xc,yc = self.c[i][0], self.c[i][1]
      dist = max(np.sqrt((xc-x)**2+(yc-y)**2) - self.r[i], 0.)
      dists.append(dist)
    return min(dists)

  def create_occupancy_grid(self):
    raise NotImplementedError

  def render(self, ax):
    for i in range(self.n):
      c = plt.Circle((self.c[i][0], self.c[i][1]), self.r[i], color='b')
      ax.add_artist(c)
    ax.plot(0.0, 0.0, 'ro')
    ax.set_xlim([-self.l, self.l])
    ax.set_ylim([-self.l, self.l])
