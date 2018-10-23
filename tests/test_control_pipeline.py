import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from systems.dubins_v1 import Dubins_v1
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from utils.fmm_map import FmmMap
from obstacles.circular_obstacle_map import CircularObstacleMap
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.goal_distance import GoalDistance
from objectives.angle_distance import AngleDistance
from objectives.objective_function import ObjectiveFunction
from trajectory.trajectory import State
from control_pipelines.control_pipeline import Control_Pipeline_v0
from utils import utils
from dotmap import DotMap


def create_params(cs, rs):
    p = DotMap()
    p.seed = 1
    p.planning_horizon_s = 1.5  # seconds
    p.n = 3
    p.map_bounds = [[-2.0, -2.0], [2.0, 2.0]]
    p.dx, p.dt = .05, .1
    p.k = int(np.ceil(p.planning_horizon_s/p.dt))

    p.lqr_coeffs = DotMap({'quad': [1.0, 1.0, 1.0, 1e-10, 1e-10],
                           'linear': [0.0, 0.0, 0.0, 0.0, 0.0]})
    p.ctrl = 1.

    p.avoid_obstacle_objective = DotMap(obstacle_margin0=0.3,
                                        obstacle_margin1=0.5,
                                        power=2,
                                        obstacle_cost=25.0)
    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=1,
                                    angle_cost=25.0)
    # Goal Distance parameters
    p.goal_distance_objective = DotMap(power=2,
                                       goal_cost=25.0,
                                       goal_margin=0.0)

    C = tf.diag(p.lqr_coeffs.quad, name='lqr_coeffs_quad')
    c = tf.constant(p.lqr_coeffs.linear,
                    name='lqr_coeffs_linear',
                    dtype=tf.float32)
    p.cost_params = {'C_gg': C, 'c_g': c}
    p.obstacle_params = {'centers_m2': cs, 'radii_m1': rs}
    p.plant_params = {'dt': p.dt}
    p.spline_params = {}
    p.control_pipeline_params = {}

    p._cost = QuadraticRegulatorRef
    p._spline = Spline3rdOrder
    p._obstacle_map = CircularObstacleMap
    p._plant = Dubins_v1
    p._control_pipeline = Control_Pipeline_v0
    return p


def test_control_pipeline(visualize=False):
    cs = np.array([[-1.0, -1.5]])
    rs = np.array([[.5]])

    p = create_params(cs, rs)
    np.random.seed(seed=p.seed)
    tf.set_random_seed(seed=p.seed)
    n = p.n
    dt = p.dx
    v0, vf = 0., 0.

    goal_pos_12 = np.array([0., 0.])[None]
    goal_pos_n2 = np.repeat(goal_pos_12, n, axis=0)

    obstacle_map = p._obstacle_map(map_bounds=p.map_bounds,
                                   **p.obstacle_params)
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
                                map_origin_2=np.array([-int(Nx/2), -int(Ny/2)]),  # lower left
                                mask_grid_mn=obstacle_occupancy_grid)

    plant = p._plant(**p.plant_params)

    obj_fn = ObjectiveFunction()

    obj_fn.add_objective(ObstacleAvoidance(
                        params=p.avoid_obstacle_objective,
                        obstacle_map=obstacle_map))
    obj_fn.add_objective(GoalDistance(
                        params=p.goal_distance_objective,
                        fmm_map=fmm_map))
    obj_fn.add_objective(AngleDistance(
                        params=p.goal_angle_objective,
                        fmm_map=fmm_map))

    # Evaluate control pipeline for given waypoints
    start_pos_nk2 = np.array([[-2., -2.],
                              [-2., -2.],
                              [-.3, 0.]], dtype=np.float32)[:, None]
    start_speed_nk1 = np.array([[v0], [v0], [0.]], dtype=np.float32)[:, None]
    start_state = State(dt, n, 1, position_nk2=start_pos_nk2,
                        speed_nk1=start_speed_nk1, variable=False)

    waypt_pos_nk2 = np.array([[-1, -.5], [-.5, -1.], [-.1, 0.]],
                             dtype=np.float32)[:, None]
    waypt_speed_nk1 = np.array([[vf], [vf], [0.]], dtype=np.float32)[:, None]
    waypt_state = State(dt, n, 1, position_nk2=waypt_pos_nk2,
                        speed_nk1=waypt_speed_nk1, variable=True)

    control_pipeline = p._control_pipeline(system_dynamics=plant,
                                           params=p,
                                           **p.control_pipeline_params)
    trajectory_lqr = control_pipeline.plan(start_state=start_state,
                                           goal_state=waypt_state)

    # Objective Value
    obj_val = obj_fn.evaluate_function(trajectory_lqr)
    obj1, obj2, obj3 = obj_val.numpy()
    val = 8333452.5
    assert(np.abs(obj2-val)/val < 1e-4)
    val = 36063.99
    assert(np.abs(obj1-val)/val < 1e-4)
    val = 0.8815087
    assert(np.abs(obj3-val)/val < 1e-4)

    if visualize:
        traj_spline = control_pipeline.traj_spline
        waypt_n5 = waypt_state.position_heading_speed_and_angular_speed_nk5()[:, 0]
        fig, _, axes = utils.subplot2(plt, (4, 2), (8, 8), (.4, .4))
        axes = axes[::-1]
        ax = axes[0]
        obstacle_map.render(ax)
        ax.set_title('Occupancy Grid')

        ax = axes[1]
        ax.contour(fmm_map.fmm_distance_map.voxel_function_mn, cmap='gray')
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 80)
        ax.set_title('Fmm Distance Map')

        wpt_13 = waypt_n5[0, :3]
        ax = axes[2]
        obstacle_map.render(ax)
        traj_spline.render(ax, batch_idx=0)
        ax.set_title('Spline, Wpt: [%.03f, %.03f, %.03f]'.format(wpt_13[0],
                                                                 wpt_13[1],
                                                                 wpt_13[2]))

        ax = axes[3]
        obstacle_map.render(ax)
        trajectory_lqr.render(ax, batch_idx=0)
        ax.set_title('LQR Traj, Cost: %.05f'.format(obj_val[0]))

        wpt_13 = waypt_n5[1, :3]
        ax = axes[4]
        obstacle_map.render(ax)
        traj_spline.render(ax, batch_idx=1)
        ax.set_title('Spline, Wpt: [%.03f, %.03f, %.03f]'.format(wpt_13[0],
                                                                 wpt_13[1],
                                                                 wpt_13[2]))

        ax = axes[5]
        obstacle_map.render(ax)
        trajectory_lqr.render(ax, batch_idx=1)
        ax.set_title('LQR Traj, Cost: %.05f'.format(obj_val[1]))

        wpt_13 = waypt_n5[2, :3]
        ax = axes[6]
        obstacle_map.render(ax)
        traj_spline.render(ax, batch_idx=2)
        ax.set_title('Spline, Wpt: [%.03f, %.03f, %.03f]'.format(wpt_13[0],
                                                                 wpt_13[1],
                                                                 wpt_13[2]))

        ax = axes[7]
        obstacle_map.render(ax)
        trajectory_lqr.render(ax, batch_idx=2)
        ax.set_title('LQR Traj, Cost: %.05f'.format(obj_val[2]))
        plt.show()
    else:
        print('Run with visualize=True to visualize the control pipeline')


if __name__ == '__main__':
    test_control_pipeline(visualize=False)
