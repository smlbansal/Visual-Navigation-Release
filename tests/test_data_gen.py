import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe
import matplotlib
import matplotlib.pyplot as plt
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from optCtrl.lqr import LQRSolver
from systems.dubins_v1 import Dubins_v1
from trajectory.spline.db_3rd_order_spline import DB3rdOrderSpline
from data_gen.data_gen import Data_Generator
from obstacles.circular_obstacle_map import CircularObstacleMap
from dotmap import DotMap

def create_params():
    p = DotMap()
    p.seed = 1
    p.n = 1
    p.k = 20
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
    params._spline = DB3rdOrderSpline
    params._obstacle_map = CircularObstacleMap
    params._plant = Dubins_v1 
    return params
 
def test_data_gen0():
    p = create_params()#load_params('v0')
    np.random.seed(seed=p.seed)
    tf.set_random_seed(seed=p.seed)
    n,k = p.n, p.k
    map_bounds = p.map_bounds
    dx, dt = p.dx, p.dt 
    v0, vf = 0., 0.
    wx = np.random.uniform(map_bounds[0][0], map_bounds[1][0])
    wy = np.random.uniform(map_bounds[0][1], map_bounds[1][1])
    wt = np.random.uniform(-np.pi, np.pi)
    
    waypt_15 = np.array([wx,wy,wt, vf, 0.])[None]
    start_15 = np.array([0., 0., 0., v0, 0.])[None]
    goal_pos_12 = np.array([0., 0.])[None]  
    
    waypt_n5 = np.repeat(waypt_15, n, axis=0)
    start_n5 = np.repeat(start_15, n, axis=0)
    goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
    
 
    cs = np.array([[-1.0, -1.5]])
    rs = np.array([[.5]])
    obj_params = create_obj_params(p, cs, rs) 

    data_gen = Data_Generator(exp_params=p,
                            obj_params=obj_params,
                            start_n5=start_n5,
                            goal_pos_n2=goal_pos_n2,
                            k=k)
    num_iter = 100
    learning_rate = 1e-4
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    waypt_n5 = tfe.Variable(waypt_n5, name='waypt', dtype=tf.float32)
    for i in range(num_iter):
        obj_val, grads, variables = data_gen.compute_obj_val_and_grad(waypt_n5)
        print('Iter: %.05f'%(obj_val))
        opt.apply_gradients(zip(grads, variables)) 
    
if __name__=='__main__':
    test_data_gen0()
