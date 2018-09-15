import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from trajectory.trajectory import Trajectory

def test_random_circular_obstacle_map():
    from obstacles.circular_obstacle_map import CircularObstacleMap
    np.random.seed(seed=1)
    dt=.1
    n,k = 100, 20
    map_bounds = [(-2,-2), (2,2)] #[(min_x, min_y), (max_x, max_y)]
    min_n, max_n = 2, 4
    min_r, max_r = .25, .5

    pos_nk2 = tf.zeros((n,k,2), dtype=tf.float32)
    trajectory = Trajectory(dt=dt, k=k, position_nk2=pos_nk2)

    grid = CircularObstacleMap.init_random_map(map_bounds, min_n, max_n, min_r, max_r)
    obs_dists_nk = grid.dist_to_nearest_obs(trajectory)
    assert(np.allclose(obs_dists_nk, np.ones((n,k))*0.32977, atol=1e-4))

    xs = np.linspace(map_bounds[0][0], map_bounds[1][0], 100, dtype=np.float32)
    ys = np.linspace(map_bounds[0][1], map_bounds[1][1], 100, dtype=np.float32)
    XS, YS = tf.meshgrid(xs, ys[::-1])
   
    occupancy_grid_nn = grid.create_occupancy_grid(XS, YS)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    grid.render(ax)

    ax = fig.add_subplot(122)
    ax.imshow(occupancy_grid_nn, cmap='gray')
    ax.set_axis_off()
    plt.show()
 
def test_circular_obstacle_map():
    from obstacles.circular_obstacle_map import CircularObstacleMap
    np.random.seed(seed=1)
    n,k = 100, 20
    dt=.1
    map_bounds = [(-2,-2), (2,2)] #[(min_x, min_y), (max_x, max_y)]

    pos_nk2 = tf.zeros((n,k,2), dtype=tf.float32)
    trajectory = Trajectory(dt=dt, k=k, position_nk2=pos_nk2)

    cs = [[-.75, .5], [0, .5], [.75, .5], [-.35, 1.], [.35, 1.]]
    rs = [[.1], [.1], [.1], [.1], [.1]]
    grid = CircularObstacleMap(map_bounds, cs, rs)
    obs_dists_nk = grid.dist_to_nearest_obs(trajectory)
    
    assert(np.allclose(obs_dists_nk, np.ones((n,k))*.4))

    xs = np.linspace(map_bounds[0][0], map_bounds[1][0], 100, dtype=np.float32)
    ys = np.linspace(map_bounds[0][1], map_bounds[1][1], 100, dtype=np.float32)
    XS, YS = tf.meshgrid(xs, ys[::-1])
   
    occupancy_grid_nn = grid.create_occupancy_grid(XS, YS)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    grid.render(ax)

    ax = fig.add_subplot(122)
    ax.imshow(occupancy_grid_nn, cmap='gray')
    ax.set_axis_off()
    plt.show()

if __name__ == '__main__':
    test_random_circular_obstacle_map()
    test_circular_obstacle_map()
