import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
from trajectory.trajectory import Trajectory
from obstacles.circular_obstacle_map import CircularObstacleMap


def test_random_circular_obstacle_map(visualize=False):
    np.random.seed(seed=1)
    dt = .1
    n, k = 100, 20

    # Obstacle map
    map_bounds = [(-2., -2.), (2., 2.)]  # [(min_x, min_y), (max_x, max_y)]
    min_n, max_n = 2, 4
    min_r, max_r = .25, .5
    grid = CircularObstacleMap.init_random_map(map_bounds,
                                               min_n, max_n, min_r, max_r)

    # Trajectory
    pos_nk2 = tf.zeros((n, k, 2), dtype=tf.float32)
    trajectory = Trajectory(dt=dt, n=n, k=k, position_nk2=pos_nk2)

    # Expected_distances
    expected_distances_m = (tf.norm(grid.obstacle_centers_m2, axis=1) -
                            grid.obstacle_radii_m1[:, 0])
    expected_min_distance = min(expected_distances_m.numpy())

    # Computed distances
    obs_dists_nk = grid.dist_to_nearest_obs(trajectory.position_nk2())

    assert(np.allclose(obs_dists_nk, np.ones((n, k))*expected_min_distance,
                       atol=1e-4))

    if visualize:
        xs = np.linspace(map_bounds[0][0], map_bounds[1][0], 100,
                         dtype=np.float32)
        ys = np.linspace(map_bounds[0][1], map_bounds[1][1], 100,
                         dtype=np.float32)
        XS, YS = tf.meshgrid(xs, ys[::-1])

        occupancy_grid_nn = grid.create_occupancy_grid(XS, YS)

        fig = plt.figure()
        ax = fig.add_subplot(121)
        grid.render(ax)

        ax = fig.add_subplot(122)
        ax.imshow(occupancy_grid_nn, cmap='gray', origin='lower')
        ax.set_axis_off()
        plt.show()
    else:
        print('rerun test_random_circular_obstacle_map with visualize=True to\
              visualize the obstacle_map')


def test_circular_obstacle_map(visualize=False):
    np.random.seed(seed=1)
    n, k = 100, 20
    dt = .1
    map_bounds = [(-2, -2), (2, 2)]  # [(min_x, min_y), (max_x, max_y)]

    pos_nk2 = tf.zeros((n, k, 2), dtype=tf.float32)
    trajectory = Trajectory(dt=dt, n=n, k=k, position_nk2=pos_nk2)

    cs = np.array([[-.75, .5], [0, .5], [.75, .5], [-.35, 1.], [.35, 1.]])
    rs = np.array([[.1], [.1], [.1], [.1], [.1]])
    grid = CircularObstacleMap(map_bounds, cs, rs)
    obs_dists_nk = grid.dist_to_nearest_obs(trajectory.position_nk2())

    assert(np.allclose(obs_dists_nk, np.ones((n, k))*.4))

    if visualize:
        xs = np.linspace(map_bounds[0][0], map_bounds[1][0], 100,
                         dtype=np.float32)
        ys = np.linspace(map_bounds[0][1], map_bounds[1][1], 100,
                         dtype=np.float32)
        XS, YS = tf.meshgrid(xs, ys[::-1])

        occupancy_grid_nn = grid.create_occupancy_grid(XS, YS)

        fig = plt.figure()
        ax = fig.add_subplot(121)
        grid.render(ax)

        ax = fig.add_subplot(122)
        ax.imshow(occupancy_grid_nn, cmap='gray', origin='lower')
        ax.set_axis_off()
        plt.show()
    else:
        print('rerun test_circular_obstacle_map with visualize=True to\
              visualize the obstacle_map')



if __name__ == '__main__':
    test_random_circular_obstacle_map(visualize=False)
    test_circular_obstacle_map(visualize=False)
