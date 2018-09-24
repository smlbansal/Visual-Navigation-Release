import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from optCtrl.lqr import LQRSolver
from systems.dubins_v1 import Dubins_v1
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from utils.fmm_map import FmmMap
from obstacles.circular_obstacle_map import CircularObstacleMap
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.goal_distance import GoalDistance
from objectives.angle_distance import AngleDistance
from objectives.objective_function import ObjectiveFunction
from utils import utils
from dotmap import DotMap

def create_params():
    p = DotMap()
    p.seed = 1
    p.n = 3
    p.k = 100
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
 
def test_control_pipeline(visualize=False):
    p = create_params()
    np.random.seed(seed=p.seed)
    tf.set_random_seed(seed=p.seed)
    n,k = p.n,p.k
    map_bounds = p.map_bounds
    dx, dt = p.dx, p.dt 
    v0, vf = 0., 0.
    
    waypt_35 = np.array([[-1., -.5 , 0., vf, 0.],[-.5, -1., 0., vf, 0.], [-.1, 0., 0., 0., 0.]])
    start_35 = np.array([[-2., -2., 0., v0, 0.],[-2., -2., 0., v0, 0.],[-.3, 0., 0., 0., 0.]])
    goal_pos_12 = np.array([0., 0.])[None]  

    waypt_n5 = waypt_35
    start_n5 = start_35
    goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
 
    cs = np.array([[-1.0, -1.5]])
    rs = np.array([[.5]])
    obj_params = create_obj_params(p, cs, rs) 

    p1, p2 = p, obj_params
    obstacle_map = p2._obstacle_map(map_bounds=p1.map_bounds, **p2.obstacle_params)
    mb = p1.map_bounds
    Nx, Ny = int((mb[1][0] - mb[0][0])/p1.dx), int((mb[1][1] - mb[0][1])/p1.dx)
    xx, yy = np.meshgrid(np.linspace(mb[0][0], mb[1][0], Nx), np.linspace(mb[0][1], mb[1][1], Ny), indexing='xy')
    obstacle_occupancy_grid = obstacle_map.create_occupancy_grid(tf.constant(xx, dtype=tf.float32),
                                                             tf.constant(yy, dtype=tf.float32))
    fmm_map = FmmMap.create_fmm_map_based_on_goal_position(goal_positions_n2=goal_pos_n2,
                                                       map_size_2=np.array([Nx, Ny]),
                                                       dx=p1.dx,
                                                       map_origin_2=np.array([-int(Nx/2), -int(Ny/2)]), #lower left
                                                       mask_grid_mn=obstacle_occupancy_grid)

    plant = p2._plant(**p2.plant_params)
    traj_spline = p2._spline(dt=p1.dt, n=n, k=k, start_n5=start_n5, **p2.spline_params) 
    cost_fn = p2._cost(trajectory_ref=traj_spline, system=plant, **p2.cost_params) 
    lqr_solver = LQRSolver(T=k-1, dynamics=plant, cost=cost_fn)
    obj_fn = ObjectiveFunction()
    
    obj_fn.add_objective(ObstacleAvoidance(params=p1.avoid_obstacle_objective, obstacle_map=obstacle_map))
    obj_fn.add_objective(GoalDistance(params=p1.goal_distance_objective, fmm_map=fmm_map))
    obj_fn.add_objective(AngleDistance(params=p1.goal_angle_objective, fmm_map=fmm_map))

    #Evaluate control pipeline for given waypoints
    waypt_n5 = tfe.Variable(waypt_n5, name='waypt', dtype=tf.float32)  
    
    #Spline
    ts_nk = tf.tile(tf.linspace(0., dt*k, k)[None], [n,1])
    traj_spline.fit(goal_n5=waypt_n5, factors_n2=None)
    traj_spline.eval_spline(ts_nk, calculate_speeds=False)
    
    #LQR
    x_nkd, u_nkf = plant.parse_trajectory(traj_spline)
    x0_n1d = x_nkd[:,0:1] 
    lqr_res = lqr_solver.lqr(x0_n1d, traj_spline, verbose=False)
    trajectory_lqr = lqr_res['trajectory_opt']
    
    #Objective Value
    obj_val = obj_fn.evaluate_function(trajectory_lqr)
    obj1, obj2, obj3 = obj_val.numpy()
    assert(obj2 > obj1 and obj1 > obj3)
   
    if visualize: 
        fig, _, axes = utils.subplot2(plt, (4,2), (8,8), (.4, .4))
        axes = axes[::-1]
        ax = axes[0]
        obstacle_map.render(ax)
        ax.set_title('Occupancy Grid')

        ax = axes[1]
        ax.contour(fmm_map.fmm_distance_map.voxel_function_mn, cmap='gray')
        ax.set_xlim(0,80)
        ax.set_ylim(0,80)
        ax.set_title('Fmm Distance Map')
      
        wpt_13 = waypt_n5[0,:3] 
        ax = axes[2]
        obstacle_map.render(ax)
        traj_spline.render(ax, batch_idx=0)
        ax.set_title('Spline, Wpt: [%.03f, %.03f, %.03f]'%(wpt_13[0], wpt_13[1], wpt_13[2])) 
         
        ax = axes[3]
        obstacle_map.render(ax)
        trajectory_lqr.render(ax, batch_idx=0)
        ax.set_title('LQR Traj, Cost: %.05f'%(obj_val[0])) 
        
        wpt_13 = waypt_n5[1,:3]
        ax = axes[4]
        obstacle_map.render(ax)
        traj_spline.render(ax, batch_idx=1)
        ax.set_title('Spline, Wpt: [%.03f, %.03f, %.03f]'%(wpt_13[0], wpt_13[1], wpt_13[2])) 
     
        ax = axes[5]
        obstacle_map.render(ax)
        trajectory_lqr.render(ax, batch_idx=1)
        ax.set_title('LQR Traj, Cost: %.05f'%(obj_val[1])) 

        wpt_13 = waypt_n5[2,:3]
        ax = axes[6]
        obstacle_map.render(ax)
        traj_spline.render(ax, batch_idx=2)
        ax.set_title('Spline, Wpt: [%.03f, %.03f, %.03f]'%(wpt_13[0], wpt_13[1], wpt_13[2])) 
     
        ax = axes[7]
        obstacle_map.render(ax)
        trajectory_lqr.render(ax, batch_idx=2)
        ax.set_title('LQR Traj, Cost: %.05f'%(obj_val[2])) 
        plt.show()
    else:
        print('Run with visualize=True to visualize the control pipeline')

if __name__=='__main__':
    test_control_pipeline(visualize=False) 
