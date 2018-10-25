from dotmap import DotMap
import numpy as np
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from obstacles.circular_obstacle_map import CircularObstacleMap
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from planners.sampling_planner import SamplingPlanner
from systems.dubins_v2 import DubinsV2
from control_pipelines.control_pipeline_v1 import Control_Pipeline_v1
from control_pipelines.control_pipeline_v0 import Control_Pipeline_v0
from simulators.circular_obstacle_map_simulator import CircularObstacleMapSimulator


def load_params():
    p = DotMap()
    p.seed = 1  # for tf and numpy seeding
    p.simulator_seed = 1
    p.n = int(1e3)  # batch size
    p.dx = 0.05  # grid discretization
    p.dt = .05  # time discretization

    # [[min_x, min_y], [max_x, max_y]]
    p.map_bounds = [[0.0, 0.0], [8.0, 8.0]]
    # in egocentric coordinates
    p.waypoint_bounds = [[-1., 0.0], [1., 1.]]

    # Horizons in seconds
    p.episode_horizon_s = 20.0
    p.planning_horizons_s = [1.5]
    p.control_horizon_s = 1.5

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

    p.lqr_quad_coeffs = np.array([1.0, 1.0, 1.0, 1e-10, 1e-10], dtype=np.float32)
    p.lqr_linear_coeffs = np.zeros((5), dtype=np.float32)

    # Store params as dictionaries so they can be used with **kwargs
    p.spline_params = {'epsilon': 1e-10}

    centers_m2 = [[2.0, 2.0]]
    radii_m1 = [[.5]]
    p.obstacle_map_params = {'centers_m2': centers_m2,
                             'radii_m1': radii_m1}

    # Based on Turtlebot parameters
    p.system_dynamics_params = {'v_bounds': [0.0, .6],
                                'w_bounds': [-1.1, 1.1]}

    precompute = True
    p.planner_params = DotMap(mode='uniform', precompute=precompute,
                              velocity_disc=.01, dx=.1,
                              num_theta_bins=11)

    p.control_pipeline_params = {'precompute': precompute,
                                 'load_from_pickle_file': True,
                                 'bin_velocity': True}

    # Simulator Params
    obstacle_map_reset_params = DotMap(reset_type='random',
                                       params={'min_n': 4, 'max_n': 7, 'min_r': .3, 'max_r': .8})
    start_config_reset_params = DotMap(reset_type='random')
    goal_config_reset_params = DotMap(reset_type='random')
    reset_params = DotMap(obstacle_map=obstacle_map_reset_params,
                          start_config=start_config_reset_params,
                          goal_config=goal_config_reset_params)

    p.simulator_params = DotMap(goal_cutoff_dist=p.goal_distance_objective.goal_margin,
                                goal_dist_norm=2,  # Default is l2 norm
                                reset_params=reset_params,
                                episode_termination_reasons=['Timeout', 'Collision', 'Success'],
                                episode_termination_colors=['b', 'r', 'g'])

    p.num_validation_goals = 50
    return p
