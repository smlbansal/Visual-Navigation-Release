import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe
import matplotlib
import matplotlib.pyplot as plt
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from optCtrl.lqr import LQRSolver
from systems.dubins_v1 import DubinsV1
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from data_gen.data_gen import Data_Generator
from obstacles.circular_obstacle_map import CircularObstacleMap
from dotmap import DotMap
from utils import utils
import os
from timeit import default_timer as timer

def create_params():
    p = DotMap()
    p.seed = 1
    p.horizon = 1.5 #seconds
    p.dx, p.dt = .05, .01
    p.k = int(np.ceil(p.horizon/p.dt))
    p.map_bounds = [[-2.0, -2.0], [2.0, 2.0]]
      
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
    params._plant = DubinsV1 
    return params

def _problem_params(problem, n):
    if problem == 0:
        cs = np.array([[-1.0, -1.5]])
        rs = np.array([[.5]])
        goal_pos_12 = np.array([0., 0.])[None]  
        goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
        return cs, rs, goal_pos_n2
    elif problem == 1:
        cs = np.array([[1.0, 1.5]])
        rs = np.array([[.5]])
        goal_pos_12 = np.array([0., 0.])[None]  
        goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
        return cs, rs, goal_pos_n2
    elif problem == 2:
        cs = np.array([[-1.0, -1.5],[-1.0, 1.5]])
        rs = np.array([[.5], [.5]])
        goal_pos_12 = np.array([0., 0.])[None]  
        goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
        return cs, rs, goal_pos_n2
    elif problem == 3:
        cs = np.array([[-1.0, -1.5],[0.0, -1.5]])
        rs = np.array([[.5], [.5]])
        goal_pos_12 = np.array([1.5, -1.0])[None] 
        goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
        return cs, rs, goal_pos_n2
    elif problem == 4:
        cs = np.array([[0.0, 0.0],[0.0, -1.5]])
        rs = np.array([[.5], [.5]])
        goal_pos_12 = np.array([1.5, -1.0])[None] 
        goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
        return cs, rs, goal_pos_n2
    elif problem == 5:
        cs = np.array([[0.0, 0.0],[0.0, -1.5]])
        rs = np.array([[.5], [.5]])
        goal_pos_12 = np.array([1.9, -1.9])[None] 
        goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
        return cs, rs, goal_pos_n2
    elif problem == 6:
        cs = np.array([[0.0, -1.0]])
        rs = np.array([[1.0]])
        goal_pos_12 = np.array([1.9, -1.9])[None] 
        goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
        return cs, rs, goal_pos_n2
    elif problem == 7:
        cs = np.array([[-1.9, 0.0], [-1.0, 0.0]])
        rs = np.array([[.25],[.25]])
        goal_pos_12 = np.array([-1.9, 1.9])[None] 
        goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
        return cs, rs, goal_pos_n2
    elif problem == 8:
        cs = np.array([[-1.9, 0.0], [-1.0, 0.0], [-1.5, 1.0]])
        rs = np.array([[.25],[.25], [.25]])
        goal_pos_12 = np.array([-1.9, 1.9])[None] 
        goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
        return cs, rs, goal_pos_n2
    elif problem == 9:
        cs = np.array([[-1.9, 0.0], [-1.0, 0.0], [-1.5, 1.0]])
        rs = np.array([[.25],[.25], [.25]])
        goal_pos_12 = np.array([-.5, 1.0])[None] 
        goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
        return cs, rs, goal_pos_n2
    else:
        assert(False)

def build_data_gen(n, problem):
    p = create_params()
    p.n=int(n)
    np.random.seed(seed=p.seed)
    tf.set_random_seed(seed=p.seed)
    n,k = p.n, p.k
    map_bounds = p.map_bounds
    dx, dt = p.dx, p.dt 
    v0, vf = 0., 0.
    wx = np.random.uniform(map_bounds[0][0], map_bounds[1][0], size=n)
    wy = np.random.uniform(map_bounds[0][1], map_bounds[1][1], size=n)
    wt = np.random.uniform(-np.pi, np.pi, size=n)
    vf = np.ones(n)*vf
    wf = np.zeros(n)
    waypt_n5 = np.stack([wx,wy,wt,vf,wf], axis=1)
    
    start_15 = np.array([-2., -2., 0., v0, 0.])[None]
    map_origin_2 = (start_15[0,:2]/dx).astype(np.int32)
    start_n5 = np.repeat(start_15, n, axis=0)
 
    cs, rs, goal_pos_n2 = _problem_params(problem=problem, n=n)
    obj_params = create_obj_params(p, cs, rs) 

    data_gen = Data_Generator(exp_params=p,
                            obj_params=obj_params,
                            start_n5=start_n5,
                            goal_pos_n2=goal_pos_n2,
                            k=k,
                            map_origin_2=map_origin_2)
    waypt_n5 = tfe.Variable(waypt_n5, name='waypt', dtype=tf.float32)
    return data_gen, waypt_n5

def random_data_gen(n=5e4, problem=0, visualize=False, fig=None, axes=None):
    start = timer()
    data_gen, waypt_n5 = build_data_gen(n=n, problem=problem)
    obj_vals = data_gen.eval_objective(waypt_n5)
    min_idx = tf.argmin(obj_vals)
    min_waypt = waypt_n5[min_idx]
    min_cost = obj_vals[min_idx]
    end = timer()
    delta_t = end-start

    if visualize:
        fig.suptitle('Random Based Opt (n=%.02e), Cost*: %.03f, Waypt*: [%.03f, %.03f, %.03f]'%(n, min_cost, min_waypt[0], min_waypt[1], min_waypt[2]))
        axes = axes[::-1]
        data_gen.render(axes, batch_idx=min_idx.numpy())
    return min_cost, delta_t

def benchmark_random_data_gen():
    ns  = [500, 1e3, 5e3, 1e4, 5e4, 1e5]
    problems = np.r_[:10]
    fig, _, axes = utils.subplot2(plt, (2,2), (8,8), (.4, .4))
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(211)
    ax2 = fig2.add_subplot(212)
    for problem in problems:
        costs, times = [], []
        problem_dir = './tmp/benchmark_random_data_gen/problem_%s'%(problem)
        utils.mkdir_if_missing(problem_dir)
        for n in ns:
            cost, delta_t = random_data_gen(n=n, problem=problem, visualize=True, fig=fig, axes=axes)
            costs.append(cost); times.append(delta_t)
            filename = os.path.join(problem_dir, 'random_opt_%d.png'%(n))
            fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=100)
        filename = os.path.join(problem_dir, 'summary.png')
        ax1.clear()
        ax1.plot(ns, costs, 'r--')
        ax1.set_title('Optimal Trajectory Cost vs # Samples')
        ax2.clear()
        ax2.plot(ns, times, 'r--')
        ax2.set_title('Runtime(s) vs # Samples')
        fig2.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=100)

def main():
    """ Benchmark the random data generation method (runtime and cost of optimal waypoint)
    with different batch sizes on 10 different test problems"""
    plt.style.use('ggplot')
    benchmark_random_data_gen()

if __name__=='__main__':
    main()
 
