import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe
import matplotlib
import matplotlib.pyplot as plt
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from obstacles.circular_obstacle_map import CircularObstacleMap
from data_gen.data_gen import Data_Generator
from systems.dubins_v1 import Dubins_v1
from optCtrl.lqr import LQRSolver
from dotmap import DotMap
from utils import utils

def create_params():
    p = DotMap()
    p.seed = 1
    p.n = 1
    p.k = 15
    p.map_bounds = [[-2.0, -2.0], [2.0, 2.0]]
    p.dx, p.dt = .05, .1
      
    p.lqr_coeffs = DotMap({'quad' : [1.0, 1.0, 1.0, 1e-10, 1e-10],
                                    'linear' : [0.0, 0.0, 0.0, 0.0, 0.0]})
    p.ctrl = 1.

    p.avoid_obstacle_objective = DotMap(obstacle_margin=0.3,
                                        power=2,
                                        obstacle_cost=25.0)
    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=1,
                                    angle_cost=25.0)
    # Goal Distance parameters
    p.goal_distance_objective = DotMap(power=2,
                                       goal_cost=25.0)

    return p 

def create_obj_params(p, cs, rs):
    C, c = tf.diag(p.lqr_coeffs.quad, name='lqr_coeffs_quad'), tf.constant(p.lqr_coeffs.linear, name='lqr_coeffs_linear', dtype=tf.float32)

    params = DotMap()
    params.cost_params = {'C' : C, 'c' : c}
    params.obstacle_params = {'centers_m2':cs, 'radii_m1':rs}
    params.plant_params = {'dt' : p.dt}
    params.spline_params = {}
     
    params._cost = QuadraticRegulatorRef
    params._spline = Spline3rdOrder
    params._obstacle_map = CircularObstacleMap
    params._plant = Dubins_v1 
    return params

def visualize_heatmap(n, theta_bins=10, plot_3d=False, log_cost=False):
    original_n = n
    p = create_params()
    np.random.seed(seed=p.seed)
    tf.set_random_seed(seed=p.seed)
    n,k = int(n), p.k
    map_bounds = p.map_bounds
    dx, dt = p.dx, p.dt 
    v0, vf = 0., 0.
    wx = np.linspace(map_bounds[0][0], map_bounds[1][0], n)
    wy = np.linspace(map_bounds[0][1], map_bounds[1][1], n)
    wt = np.linspace(-np.pi, np.pi, theta_bins)
    wx, wy, wt = np.meshgrid(wx,wy,wt)
    wx, wy, wt = wx.ravel(), wy.ravel(), wt.ravel()
    n=len(wx)
    p.n=n
    vf = np.ones(n)*vf
    wf = np.zeros(n)
    
    start_15 = np.array([-2., -2., 0., v0, 0.])[None]
    map_origin_2 = (start_15[0,:2]/dx).astype(np.int32)
    goal_pos_12 = np.array([0., 0.])[None]  
    
    waypt_n5 = np.stack([wx,wy,wt,vf,wf], axis=1)
    start_n5 = np.repeat(start_15, n, axis=0)
    goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
   
 
    cs = np.array([[-1.0, -1.5]])
    rs = np.array([[.5]])
    obj_params = create_obj_params(p, cs, rs) 

    data_gen = Data_Generator(exp_params=p,
                            obj_params=obj_params,
                            start_n5=start_n5,
                            goal_pos_n2=goal_pos_n2,
                            k=k,
                            map_origin_2=map_origin_2)
    waypt_n5 = tfe.Variable(waypt_n5, name='waypt', dtype=tf.float32)

    obj_vals = data_gen.eval_objective(waypt_n5)
    min_idx = tf.argmin(obj_vals)
    min_waypt = waypt_n5[min_idx]
    min_cost = obj_vals[min_idx]
    print('Cost*: %.03f, Waypt*: [%.03f, %.03f, %.03f]'%(min_cost, min_waypt[0], min_waypt[1], min_waypt[2]))

    if plot_3d:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        p = ax.scatter(wx, wy, wt, c=np.log(obj_vals.numpy()), cmap=cm.hot)
        ax.set_xlabel('Wpt_X')
        ax.set_ylabel('Wpt_Y')
        ax.set_zlabel('Wpt_theta')
        ax.set_title('Log Waypt Cost, Cost*: %.03f, Waypt*: [%.03f, %.03f, %.03f]'%(min_cost, min_waypt[0], min_waypt[1], min_waypt[2]))
        fig.colorbar(p)
    else:
        sqrt_num_plots = int(np.ceil(np.sqrt(theta_bins+1)))
        fig, _, axes = utils.subplot2(plt, (sqrt_num_plots,sqrt_num_plots), (8,8), (.4, .4))
        ax = axes.pop()
        data_gen.obstacle_map.render(ax)
        axes = axes[::-1]
        for i in range(theta_bins):
            ax = axes[i]
            theta = wt[i::theta_bins]
            assert((theta == theta[0]).all())
            heatmap = tf.reshape(obj_vals[i::theta_bins], (original_n, original_n))
            if log_cost:
                heatmap = np.log(heatmap)
            p = ax.imshow(heatmap, cmap='hot', origin='lower')
            ax.set_title('Heatmap Theta = %.03f'%(theta[0]))
            fig.colorbar(p, ax=ax)
    plt.show()

def main():
    plt.style.use('ggplot')
    visualize_heatmap(n=80, theta_bins=21, plot_3d=False)

if __name__=='__main__':
    main()
 
