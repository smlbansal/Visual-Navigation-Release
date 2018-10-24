from dotmap import DotMap
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from obstacles.circular_obstacle_map import CircularObstacleMap
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from planners.sampling_planner_v2 import SamplingPlanner_v2
from systems.dubins_v2 import Dubins_v2
from control_pipelines.control_pipeline_v1 import Control_Pipeline_v1
from simulators.circular_obstacle_map_simulator import CircularObstacleMapSimulator
import utils.utils as utils
from trajectory.trajectory import State
import matplotlib.pyplot as plt
import os


def load_params():
    p = DotMap()
    p.seed = 1  # for tf and numpy seeding
    p.simulator_seed = 1
    p.n = int(1e3)  # batch size
    p.dx = 0.05  # grid discretization

    # [[min_x, min_y], [max_x, max_y]]
    p.map_bounds = [[0.0, 0.0], [8.0, 8.0]]
    # in egocentric coordinates
    p.waypoint_bounds = [[0.0, -1.0], [1., 1.]]

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
    p._system_dynamics = Dubins_v2
    p._planner = SamplingPlanner_v2
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
    p.simulator_params = {'goal_cutoff_dist': p.goal_distance_objective.goal_margin,
                          'goal_dist_norm': 2,  # Default is l2 norm
                          'end_episode_on_collision': False,
                          'end_episode_on_success': True}

    p.control_validation_params = DotMap(num_tests_per_map=1,
                                         num_maps=50)
    return p


def sample_waypoints(p, vf=0.):
    n = p.n
    wx = np.linspace(*p.planner_params['waypt_x_params'], dtype=np.float32)
    wy = np.linspace(*p.planner_params['waypt_y_params'], dtype=np.float32)
    wt = np.linspace(*p.planner_params['waypt_theta_params'], dtype=np.float32)
    wx, wy, wt = np.meshgrid(wx, wy, wt)
    wx_n11 = wx.ravel()[:, None, None]
    wy_n11 = wy.ravel()[:, None, None]
    wt_n11 = wt.ravel()[:, None, None]

    wx_n11, wy_n11, wt_n11 = p._spline.ensure_goals_valid(0.0, 0.0, wx_n11,
                                                          wy_n11, wt_n11,
                                                          epsilon=p.spline_params['epsilon'])

    vf = tf.ones((n, 1, 1), dtype=tf.float32)*vf
    waypt_pos_n12 = tf.concat([wx_n11, wy_n11], axis=2)
    waypt_egocentric_state_n = State(dt=p.dt, n=n, k=1,
                                     position_nk2=waypt_pos_n12,
                                     speed_nk1=vf,
                                     heading_nk1=wt_n11,
                                     variable=True)
    return waypt_egocentric_state_n


def plot_pipeline(pipeline, axess, fig, v0):
    logdir = '/'.join(pipeline._data_file_name().split('/')[:-1])
    logdir = os.path.join(logdir, 'plots', 'v0_{:.3f}'.format(v0))
    utils.mkdir_if_missing(logdir)
    for idx in pipeline.valid_idxs:
        pipeline.traj_spline.render(axess[:3], batch_idx=idx, freq=4, plot_control=True)
        pipeline.traj_opt.render(axess[3:], batch_idx=idx, freq=4, plot_control=True,
                                 label_start_and_end=True, name='LQR')
        filename = os.path.join(logdir, 'idx_{:d}.png'.format(idx))
        fig.savefig(filename)


def main():
    plt.style.use('ggplot')
    p = load_params()
    system_dynamics = p._system_dynamics(dt=p.dt, **p.system_dynamics_params)
    delta_v = system_dynamics.v_bounds[1] - system_dynamics.v_bounds[0]
    start_velocities = np.linspace(system_dynamics.v_bounds[0],
                                   system_dynamics.v_bounds[1],
                                   int(np.ceil(delta_v/p.planner_params['velocity_disc'])))

    waypt_egocentric_state_n = sample_waypoints(p)
    print('Control_Pipeline: {:s}'.format(p._control_pipeline.pipeline_name))
    fig, _, axs = utils.subplot2(plt, (2, 3), (8, 8), (.4, .4))
    axs = axs[::-1]
    for k in p.ks:
        for velocity in start_velocities:
            start_state = system_dynamics.init_egocentric_robot_state(dt=p.dt, n=p.n,
                                                                  v=velocity, w=0.0)
            pipeline = p._control_pipeline(system_dynamics=system_dynamics, params=p, v0=velocity,
                                           k=k, ** p.control_pipeline_params)
            pipeline.plan(start_state, waypt_egocentric_state_n)
            plot_pipeline(pipeline, axs, fig, velocity)
            num_good_waypts = pipeline.valid_idxs.shape[0].value
            percent_good_waypt = num_good_waypts/p.n
            print('k: {:d}, v0: {:.3f}, # Good Waypts: {:d}, % Good waypoints: {:.3f}'.format(k, velocity,
                                                                         num_good_waypts, percent_good_waypt))


if __name__ == '__main__':
    main()
