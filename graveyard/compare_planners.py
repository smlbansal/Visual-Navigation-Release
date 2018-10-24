import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from obstacles.circular_obstacle_map import CircularObstacleMap
from systems.dubins_v1 import Dubins_v1
from planners.sampling_planner import SamplingPlanner
from planners.gradient_planner import GradientPlanner
from dotmap import DotMap
from utils import utils
from utils.fmm_map import FmmMap
from trajectory.trajectory import SystemConfig
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.goal_distance import GoalDistance
from objectives.angle_distance import AngleDistance
from objectives.objective_function import ObjectiveFunction
from control_pipelines.control_pipeline import Control_Pipeline_v0

def create_params(planner):
    p = DotMap()
    p.seed = 1
    p.dx, p.dt = .05, .01

    # Horizons in seconds
    p.planning_horizon_s = 1.5  # .15

    # Horizons in timesteps
    p.k = int(np.ceil(p.planning_horizon_s/p.dt))

    p.map_bounds = [[-2.0, -2.0], [2.0, 2.0]]
    p.waypoint_bounds = [[-2.0, -2.0], [2.0, 2.0]]
    p.lqr_coeffs = DotMap({'quad': [1.0, 1.0, 1.0, 1e-10, 1e-10],
                           'linear': [0.0, 0.0, 0.0, 0.0, 0.0]})

    p.avoid_obstacle_objective = DotMap(obstacle_margin0=0.3,
                                        obstacle_margin1=.5,
                                        power=3,
                                        obstacle_cost=1.0)
    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=1,
                                    angle_cost=.004)
    # Goal Distance parameters
    p.goal_distance_objective = DotMap(power=2,
                                       goal_cost=.04,
                                       goal_margin=0.0)
    C = tf.diag(p.lqr_coeffs.quad, name='lqr_coeffs_quad')
    c = tf.constant(p.lqr_coeffs.linear, name='lqr_coeffs_linear',
                    dtype=tf.float32)

    p.cost_params = {'C_gg': C, 'c_g': c}
    p.spline_params = {'epsilon': 1e-10}

    p._cost = QuadraticRegulatorRef
    p._spline = Spline3rdOrder
    p._obstacle_map = CircularObstacleMap
    p._system_dynamics = Dubins_v1
    p._control_pipeline = Control_Pipeline_v0

    if planner == 'sampling':
        dx, num_theta_bins = .1, 21
        x0, y0 = p.waypoint_bounds[0]
        xf, yf = p.waypoint_bounds[1]
        nx = int((xf-x0)/dx)
        ny = int((yf-y0)/dx)
        p.n = nx*ny*num_theta_bins
        p.planner_params = {'mode': 'random',
                            'dx': dx,
                            'num_theta_bins': num_theta_bins}
        p._planner = SamplingPlanner
    elif planner == 'gradient':
        p.planner_params = {'learning_rate': 1e-2,
                            'optimizer': tf.train.AdamOptimizer,
                            'num_opt_iters': 30}
        p.n = 1
        p._planner = GradientPlanner
    else:
        assert(False)
    return p


def build_fmm_map(obstacle_map, map_origin_2, goal_pos_n2, p):
    mb = p.map_bounds
    Nx, Ny = int((mb[1][0] - mb[0][0])/p.dx), int((mb[1][1] - mb[0][1])/p.dx)
    xx, yy = np.meshgrid(np.linspace(mb[0][0], mb[1][0], Nx),
                         np.linspace(mb[0][1], mb[1][1], Ny),
                         indexing='xy')
    obstacle_occupancy_grid = obstacle_map.create_occupancy_grid(
                                        tf.constant(xx, dtype=tf.float32),
                                        tf.constant(yy, dtype=tf.float32))
    fmm_map = FmmMap.create_fmm_map_based_on_goal_position(
                        goal_positions_n2=goal_pos_n2,
                        map_size_2=np.array([Nx, Ny]),
                        dx=p.dx,
                        map_origin_2=map_origin_2,
                        mask_grid_mn=obstacle_occupancy_grid)
    return fmm_map


def build_planner(planner):
    p = create_params(planner=planner)
    np.random.seed(seed=p.seed)
    tf.set_random_seed(seed=p.seed)
    n = p.n
    dx = p.dx
    v0 = 0.

    start_5 = np.array([-2., -2., 0., v0, 0.])
    start_pos_112 = np.array([[start_5[0], start_5[1]]], dtype=np.float32)[:, None]
    start_speed_111 = np.ones((1, 1, 1), dtype=np.float32)*v0
    start_config = SystemConfig(dt=p.dt, n=1, k=1, position_nk2=start_pos_112,
                               speed_nk1=start_speed_111, variable=False)

    map_origin_2 = (start_5[:2]/dx).astype(np.int32)
    goal_pos_12 = np.array([0., 0.])[None]
    goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)

    cs = np.array([[-1.0, -1.5]])
    rs = np.array([[.5]])

    obstacle_map = p._obstacle_map(map_bounds=p.map_bounds,
                                   centers_m2=cs,
                                   radii_m1=rs)
    fmm_map = build_fmm_map(obstacle_map, map_origin_2, goal_pos_n2, p)
    system_dynamics = p._system_dynamics(dt=p.dt)

    obj_fn = ObjectiveFunction()

    if not p.avoid_obstacle_objective.empty():
        obj_fn.add_objective(ObstacleAvoidance(
                             params=p.avoid_obstacle_objective,
                             obstacle_map=obstacle_map))
    if not p.goal_distance_objective.empty():
        obj_fn.add_objective(GoalDistance(
                             params=p.goal_distance_objective,
                             fmm_map=fmm_map))
    if not p.goal_angle_objective.empty():
        obj_fn.add_objective(AngleDistance(
                             params=p.goal_angle_objective,
                             fmm_map=fmm_map))

    return p._planner(system_dynamics=system_dynamics,
                      obj_fn=obj_fn, params=p, **p.planner_params), start_config, obstacle_map, fmm_map, p


def test_sampling_planner(visualize=False):
    planner, start_config, obstacle_map, fmm_map, params = build_planner(planner='sampling')
    min_waypt, min_traj, min_cost = planner.optimize(start_config)
    if visualize:
        waypt = min_waypt.position_and_heading_nk3()[0, 0]
        fig, _, axes = utils.subplot2(plt, (2, 2), (8, 8), (.4, .4))
        fig.suptitle('Random Based Opt (n=%.02e), Cost*: %.03f, Waypt*: [%.03f, %.03f, %.03f]'%
                     (params.n, min_cost, waypt[0], waypt[1], waypt[2]))
        axes = axes[::-1]
        axs = axes[:2]
        planner.render(axs, start_config, min_waypt, obstacle_map=obstacle_map)
        plt.show()
    else:
        print('rerun test_random_based_data_gen with '
              'visualize=True to see visualization')


def test_gradient_planner(visualize=False):
    planner, start_config, obstacle_map, fmm_map, params = build_planner(planner='gradient')
    min_waypt, min_traj, min_cost = planner.optimize(start_config)
    if visualize:
        waypt = min_waypt.position_and_heading_nk3()[0, 0]
        fig, _, axes = utils.subplot2(plt, (2, 2), (8, 8), (.4, .4))
        fig.suptitle('Gradient Based Opt, Cost*: %.03f, Waypt*: [%.03f, %.03f, %.03f]'%
                     (min_cost, waypt[0], waypt[1], waypt[2]))
        axes = axes[::-1]
        axs = axes[:3]
        planner.render(axs, start_config, min_waypt, obstacle_map=obstacle_map)
        plt.savefig('./tmp/gradient_planner.png')
    else:
        print('rerun test_gradient_based_data_gen with '
              'visualize=True to see visualization')


def main():
    plt.style.use('ggplot')
    #test_sampling_planner(visualize=True)
    test_gradient_planner(visualize=True)


if __name__ == '__main__':
    main()
