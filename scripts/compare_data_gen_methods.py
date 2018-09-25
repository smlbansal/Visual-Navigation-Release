import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe
import matplotlib
import matplotlib.pyplot as plt
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from obstacles.circular_obstacle_map import CircularObstacleMap
from systems.dubins_v1 import Dubins_v1
from optCtrl.lqr import LQRSolver
from planners.sampling_planner import SamplingPlanner
from planners.gradient_planner import GradientPlanner
from dotmap import DotMap
from utils import utils
from utils.fmm_map import FmmMap
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.goal_distance import GoalDistance
from objectives.angle_distance import AngleDistance
from objectives.objective_function import ObjectiveFunction


def create_params(planner):
    p = DotMap()
    p.seed = 1
    p.horizon = 1.5#seconds
    p.dx, p.dt = .05, .1
    p.k = int(np.ceil(p.horizon/p.dt))
    p.map_bounds = [[-2.0, -2.0], [2.0, 2.0]]
    p.waypoint_bounds = [[-2.0, -2.0], [2.0, 2.0]] 
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
    C, c = tf.diag(p.lqr_coeffs.quad, name='lqr_coeffs_quad'), tf.constant(p.lqr_coeffs.linear, name='lqr_coeffs_linear', dtype=tf.float32)

    p.cost_params = {'C' : C, 'c' : c}
    p.spline_params = {}
     
    p._cost = QuadraticRegulatorRef
    p._spline = Spline3rdOrder
    p._obstacle_map = CircularObstacleMap
    p._system_dynamics = Dubins_v1
    
    if planner == 'sampling':
        dx, num_theta_bins = .1, 21
        x0, y0 = p.waypoint_bounds[0]
        xf, yf = p.waypoint_bounds[1]
        nx = int((xf-x0)/dx)
        ny = int((yf-y0)/dx)
        p.n = nx*ny*num_theta_bins
        p.planner_params = {'mode':'uniform', 'dx':dx, 'num_theta_bins': num_theta_bins}
        p._planner = SamplingPlanner
    elif planner == 'gradient':
        p.planner_params = {'learning_rate':1e-1, 
                            'optimizer':tf.train.AdamOptimizer,
                            'num_opt_iters':30}
        p.n = 1
        p._planner = GradientPlanner
    else:
        assert(False)
    return p 

def build_fmm_map(obstacle_map, map_origin_2, goal_pos_n2, p):
    mb = p.map_bounds
    Nx, Ny = int((mb[1][0] - mb[0][0])/p.dx), int((mb[1][1] - mb[0][1])/p.dx)
    xx, yy = np.meshgrid(np.linspace(mb[0][0], mb[1][0], Nx), np.linspace(mb[0][1], mb[1][1], Ny), indexing='xy')
    obstacle_occupancy_grid = obstacle_map.create_occupancy_grid(tf.constant(xx, dtype=tf.float32),
                                                             tf.constant(yy, dtype=tf.float32))
    fmm_map = FmmMap.create_fmm_map_based_on_goal_position(goal_positions_n2=goal_pos_n2,
                                                       map_size_2=np.array([Nx, Ny]),
                                                       dx=p.dx,
                                                       map_origin_2=map_origin_2,
                                                       mask_grid_mn=obstacle_occupancy_grid)
    return fmm_map

def build_planner(planner):
    p = create_params(planner=planner)
    np.random.seed(seed=p.seed)
    tf.set_random_seed(seed=p.seed)
    n,k = p.n, p.k
    map_bounds = p.map_bounds
    dx, dt = p.dx, p.dt 
    v0, vf = 0., 0.
        
    start_15 = np.array([-2., -2., 0., v0, 0.])[None]
    map_origin_2 = (start_15[0,:2]/dx).astype(np.int32)
    goal_pos_12 = np.array([0., 0.])[None]  
    
    start_n5 = np.repeat(start_15, n, axis=0)
    goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)
 
    cs = np.array([[-1.0, -1.5]])
    rs = np.array([[.5]])

    obstacle_map = p._obstacle_map(map_bounds=p.map_bounds, centers_m2=cs, radii_m1=rs)
    fmm_map = build_fmm_map(obstacle_map, map_origin_2, goal_pos_n2, p)
    system_dynamics = p._system_dynamics(dt=p.dt)

    obj_fn = ObjectiveFunction()

    if not p.avoid_obstacle_objective.empty():        
        obj_fn.add_objective(ObstacleAvoidance(params=p.avoid_obstacle_objective, obstacle_map=obstacle_map))
    if not p.goal_distance_objective.empty():
        obj_fn.add_objective(GoalDistance(params=p.goal_distance_objective, fmm_map=fmm_map))
    if not p.goal_angle_objective.empty():
        obj_fn.add_objective(AngleDistance(params=p.goal_angle_objective, fmm_map=fmm_map))

    return p._planner(system_dynamics=system_dynamics,
                obj_fn=obj_fn, params=p, start_n5=start_n5, **p.planner_params), \
                obstacle_map, fmm_map, p
 
def test_sampling_planner(visualize=False):
    planner, obstacle_map, fmm_map, params = build_planner(planner='sampling') 
    min_waypt, min_cost = planner.optimize()
    if visualize:
        fig, _, axes = utils.subplot2(plt, (2,2), (8,8), (.4, .4))
        fig.suptitle('Random Based Opt (n=%.02e), Cost*: %.03f, Waypt*: [%.03f, %.03f, %.03f]'%(params.n, min_cost, min_waypt[0], min_waypt[1], min_waypt[2]))
        axes = axes[::-1]
        planner.render(axes, min_waypt, obstacle_map=obstacle_map)
        plt.show()
    else:
        print('rerun test_random_based_data_gen with visualize=True to see visualization')

def test_gradient_planner(visualize=False):
    planner, obstacle_map, fmm_map, params = build_planner(planner='gradient') 
    min_waypt, min_cost = planner.optimize()
        
    if visualize:
        fig, _, axes = utils.subplot2(plt, (3,3), (8,8), (.4, .4))
        fig.suptitle('Gradient Based Opt, Cost*: %.03f, Waypt*: [%.03f, %.03f, %.03f]'%(min_cost, min_waypt[0], min_waypt[1], min_waypt[2]))
        axes = axes[::-1]
        ax4, axes = axes[:4], axes[4:]
        planner.render(ax4, min_waypt, obstacle_map=obstacle_map)
        plt.show()
    else:
        print('rerun test_gradient_based_data_gen with visualize=True to see visualization')
    
def main():
    plt.style.use('ggplot')
    test_sampling_planner(visualize=True)
    test_gradient_planner(visualize=True)

if __name__=='__main__':
    main()
    
