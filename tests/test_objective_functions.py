import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from obstacles.circular_obstacle_map import CircularObstacleMap
from objectives.goal_distance import GoalDistance
from objectives.objective_function import ObjectiveFunction
from trajectory.trajectory import Trajectory
from utils.fmm_map import FmmMap
from utils.utils import load_params

def test_goal_distance():
    p = load_params('v0')
    np.random.seed(seed=p.seed)
    n,k = p.n, p.k
    map_bounds = p.map_bounds
    dx = p.dx 
    
    Nx = int((map_bounds[1][0] - map_bounds[0][0])/dx + 1)
    Ny = int((map_bounds[1][1] - map_bounds[0][1])/dx + 1)

    
    assert(p.obstacle_map.name == 'circular')
    cs = p.obstacle_map.centers
    rs = p.obstacle_map.radii
    grid = CircularObstacleMap(map_bounds, cs, rs)
    
    xs = np.linspace(map_bounds[0][0], map_bounds[1][0], Nx, dtype=np.float32)
    ys = np.linspace(map_bounds[0][1], map_bounds[1][1], Ny, dtype=np.float32)
    XS, YS = tf.meshgrid(xs, ys[::-1])
    occupancy_grid_nn = grid.create_occupancy_grid(XS, YS)
  
    map_origin_2 = tf.constant([0., 0.], dtype=tf.float32)
    goalx_n1, goaly_n1 = np.ones((n,1), dtype=np.float32)*p.goal.x, np.ones((n,1), dtype=np.float32)*p.goal.y
    goal_n2 = np.concatenate([goalx_n1, goaly_n1], axis=1)
    map_size_2 = [Nx, Ny]
    fmm_map = FmmMap.create_fmm_map_based_on_goal_position(goal_n2, map_size_2, dx=p.dx,
                                                map_origin_2=map_origin_2, mask_grid_mn=occupancy_grid_nn) 
    
    goal_obj = GoalDistance(p, fmm_map)
    obj = ObjectiveFunction()
    obj.add_objective(goal_obj)
    
    x_nk1, y_nk1 = tf.ones((n,k,1), dtype=tf.float32)*1., 2.*tf.ones((n,k,1), dtype=tf.float32)
    pos_nk2 = tf.concat([x_nk1, y_nk1], axis=2)
    trajectory = Trajectory(dt=p.dt, k=k, position_nk2=pos_nk2)
    obj_n = obj.evaluate_function(trajectory)
    goal_nk = goal_obj.evaluate_objective(trajectory)

    x_nk1, y_nk1 = tf.ones((n,k,1), dtype=tf.float32)*0., 1.*tf.ones((n,k,1), dtype=tf.float32)
    pos_nk2 = tf.concat([x_nk1, y_nk1], axis=2)
    trajectory = Trajectory(dt=p.dt, k=k, position_nk2=pos_nk2)
    obj_n = obj.evaluate_function(trajectory)
    goal_nk = goal_obj.evaluate_objective(trajectory)
    #assert(np.allclose(goal_nk, np.ones((n,k))*1.))
    #assert(np.allclose(obj_n, np.ones((n))*1.))

    x_nk1, y_nk1 = tf.ones((n,k,1), dtype=tf.float32)*1., 3.*tf.ones((n,k,1), dtype=tf.float32)
    pos_nk2 = tf.concat([x_nk1, y_nk1], axis=2)
    trajectory = Trajectory(dt=p.dt, k=k, position_nk2=pos_nk2)
    obj_n = obj.evaluate_function(trajectory)
    goal_nk = goal_obj.evaluate_objective(trajectory)

    #TODO: Add some asserts here
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(occupancy_grid_nn, cmap='gray')
    ax.set_title('Occupancy Grid (Mask)')

    ax = fig.add_subplot(122)
    ax.contour(fmm_map.fmm_distance_map.voxel_function_mn)
    ax.set_title('Fmm Map')

    plt.show()

if __name__ == '__main__':
    test_goal_distance()
