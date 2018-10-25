from dotmap import DotMap
import tensorflow as tf
import numpy as np
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from obstacles.circular_obstacle_map import CircularObstacleMap
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from planners.sampling_planner import SamplingPlanner
from systems.dubins_v2 import DubinsV2
from control_pipelines.control_pipeline_v1 import Control_Pipeline_v1
from control_pipelines.control_pipeline_v0 import Control_Pipeline_v0
from simulators.circular_obstacle_map_simulator import CircularObstacleMapSimulator
import utils.utils as utils


def load_params():
    p = DotMap()
    p.seed = 1  # for tf and numpy seeding
    p.simulator_seed = 1
    p.n = int(1e3)  # batch size
    p.dx = 0.05  # grid discretization

    # [[min_x, min_y], [max_x, max_y]]
    p.map_bounds = [[0.0, 0.0], [8.0, 8.0]]
    # in egocentric coordinates
    p.waypoint_bounds = [[-1., 0.0], [1., 1.]]

    # Map Origin and size
    origin_x = int(p.map_bounds[0][0]/p.dx)
    origin_y = int(p.map_bounds[0][1]/p.dx)
    p.map_origin_2 = np.array([origin_x, origin_y], dtype=np.int32)
    Nx = int((p.map_bounds[1][0] - p.map_bounds[0][0])/p.dx)
    Ny = int((p.map_bounds[1][1] - p.map_bounds[0][1])/p.dx)
    p.map_size_2 = [Nx, Ny]

    p.dt = .05  # time discretization

    # Horizons in seconds
    p.episode_horizon_s = 20
    p.planning_horizons_s = [3.0]
    p.control_horizon_s = 20.0

    # Horizons in timesteps
    p.episode_horizon = int(np.ceil(p.episode_horizon_s/p.dt))
    p.ks = [int(np.ceil(x/p.dt)) for x in p.planning_horizons_s]
    p.control_horizon = int(np.ceil(p.control_horizon_s/p.dt))

    # Obstacle Avoidance Objective
    p.avoid_obstacle_objective = DotMap(obstacle_margin0=0.3,
                                        obstacle_margin1=0.5,
                                        power=3,
                                        obstacle_cost=1.0)
    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=1,
                                    angle_cost=.008)
    # Goal Distance parameters
    p.goal_distance_objective = DotMap(power=2,
                                       goal_cost=.08,
                                       goal_margin=.3)

    p._cost = QuadraticRegulatorRef
    p._spline = Spline3rdOrder
    p._obstacle_map = CircularObstacleMap
    p._system_dynamics = DubinsV2
    p._planner = SamplingPlanner
    p._control_pipeline = Control_Pipeline_v1
    p._simulator = CircularObstacleMapSimulator

    # Store params as dictionaries so they can be used with **kwargs
    lqr_quad_coeffs = np.array([1.0, 1.0, 1.0, 1e-10, 1e-10], dtype=np.float32)
    lqr_linear_coeffs = np.zeros((5), dtype=np.float32)
    C = tf.diag(lqr_quad_coeffs, name='lqr_coeffs_quad')
    c = tf.constant(lqr_linear_coeffs, name='lqr_coeffs_linear',
                    dtype=tf.float32)
    p.cost_params = {'C_gg': C, 'c_g': c}

    p.spline_params = {'epsilon': 1e-10}

    centers_m2 = [[2.0, 2.0]]
    radii_m1 = [[.5]]
    p.obstacle_map_params = {'centers_m2': centers_m2,
                             'radii_m1': radii_m1}

    # Based on Turtlebot parameters
    p.system_dynamics_params = {'v_bounds': [0.0, .6],
                                'w_bounds': [-1.1, 1.1]}

    # dx and num_theta_bins only used if sampling mode is uniform
    dx = .1
    num_theta_bins = utils.ensure_odd(21)
    precompute = True
    velocity_disc = .01  # discretization of velocity for control pipeline
    p.planner_params = {'mode': 'uniform',
                        'precompute': precompute,
                        'velocity_disc': velocity_disc}

    # Check implied batch size for uniform sampling
    if p.planner_params['mode'] == 'uniform':
        x0, y0 = p.waypoint_bounds[0]
        xf, yf = p.waypoint_bounds[1]
        # Make sure these are odd so the origin is included (for turning waypoints)
        nx = utils.ensure_odd(int((xf-x0)/dx))
        ny = utils.ensure_odd(int((yf-y0)/dx))
        p.planner_params['waypt_x_params'] = [x0, xf, nx]
        p.planner_params['waypt_y_params'] = [y0, yf, ny]
        p.planner_params['waypt_theta_params'] = [-np.pi/2, np.pi/2, num_theta_bins]
        p.n = int(nx*ny*num_theta_bins)

    p.control_pipeline_params = {'precompute': precompute,
                                 'load_from_pickle_file': True,
                                 'bin_velocity': True}

    # Simulator Reset Params
    obstacle_map_reset_params = DotMap(reset_type='random',
                                       params={'min_n': 4, 'max_n': 7, 'min_r': .3, 'max_r': .8})
    start_config_reset_params = DotMap(reset_type='random')
    goal_config_reset_params = DotMap(reset_type='random')
    reset_params = DotMap(obstacle_map=obstacle_map_reset_params,
                          start_config=start_config_reset_params,
                          goal_config=goal_config_reset_params)

    p.simulator_params = DotMap(goal_cutoff_dist=p.goal_distance_objective.goal_margin,
                          goal_dist_norm=2,  # Default is l2 norm
                          end_episode_on_collision=False,
                          end_episode_on_success=True,
                          reset_params=reset_params)

    p.control_validation_params = DotMap(num_tests_per_map=1,
                                         num_maps=50)
    return p
