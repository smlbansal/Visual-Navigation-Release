import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from systems.dubins_v3 import Dubins_v3
from systems.dubins_v2 import Dubins_v2
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from trajectory.spline.spline_3rd_order_turn_v1 import Spline3rdOrderTurnV1
from trajectory.spline.spline_3rd_order_turn_v2 import Spline3rdOrderTurnV2
from trajectory.trajectory import State
from control_pipelines.control_pipeline import Control_Pipeline_v0
from utils import utils
from dotmap import DotMap


def create_params():
    p = DotMap()
    p.seed = 1
    p.planning_horizon_s = 1.5  # seconds
    p.n = 1
    p.map_bounds = [[-2.0, -2.0], [2.0, 2.0]]
    p.dx, p.dt = .05, .1
    p.k = int(np.ceil(p.planning_horizon_s/p.dt))

    p._plant = Dubins_v3
    if p._plant is Dubins_v3:
        p.lqr_coeffs = DotMap({'quad': [1.0, 1.0, 1.0, 1e-5, 1e-5, 1e-5, 1e-5],
                               'linear': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
    else:
        p.lqr_coeffs = DotMap({'quad': [1.0, 1.0, 1.0, 1e-5, 1e-5],
                               'linear': [0.0, 0.0, 0.0, 0.0, 0.0]})
    p.ctrl = 1.

    C = tf.diag(p.lqr_coeffs.quad, name='lqr_coeffs_quad')
    c = tf.constant(p.lqr_coeffs.linear,
                    name='lqr_coeffs_linear',
                    dtype=tf.float32)
    p.cost_params = {'C_gg': C, 'c_g': c}
    p.plant_params = {'dt': p.dt}
    p.spline_params = {}
    p.control_pipeline_params = {}

    p._cost = QuadraticRegulatorRef
    #p._spline = Spline3rdOrderTurnV2
    p._spline = Spline3rdOrder
    p._control_pipeline = Control_Pipeline_v0
    return p


def visualize_control_pipeline_turn():
    epsilon = 0.
    p = create_params()
    np.random.seed(seed=p.seed)
    tf.set_random_seed(seed=p.seed)
    n = p.n
    dt = p.dx

    theta_goal = np.pi/2.
    v0 = np.random.uniform(0., 0.5, 1)[0]  # Initial speed
    v0 = .6
    print(v0)
    vf = 0.

    # Initial State is [0, 0, 0, v0, 0]
    x0, y0 = -1., -1.
    start_pos_nk2 = np.array([[[x0, y0]]])
    start_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32)*v0

    goal_pos_nk2 = tf.zeros((n, 1, 2), dtype=tf.float32) + epsilon
    goal_heading_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * theta_goal
    goal_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * vf

    start_state = State(dt, n, 1, position_nk2=start_pos_nk2, speed_nk1=start_speed_nk1, variable=False)
    goal_state = State(dt, n, 1, position_nk2=goal_pos_nk2,
                       speed_nk1=goal_speed_nk1, heading_nk1=goal_heading_nk1,
                       variable=True)

    plant = p._plant(**p.plant_params)
    control_pipeline = p._control_pipeline(system_dynamics=plant,
                                           params=p,
                                           **p.control_pipeline_params)
    trajectory_lqr = control_pipeline.plan(start_state=start_state,
                                           goal_state=goal_state)

    traj_spline = control_pipeline.traj_spline
    waypt_n5 = goal_state.position_heading_speed_and_angular_speed_nk5()[:, 0]
    fig, _, axes = utils.subplot2(plt, (4, 2), (8, 8), (.4, .4))
    axes = axes[::-1]

    wpt_13 = waypt_n5[0, :3]
    ax = axes[2]
    traj_spline.render(ax, batch_idx=0, freq=1)
    ax.set_title('Spline, Wpt: [{:e}, {:e}, {:.03f}]'.format(wpt_13[0],
                                                                  wpt_13[1],
                                                                  wpt_13[2]))

    ax = axes[3]
    trajectory_lqr.render(ax, batch_idx=0, freq=1)
    end_pose_3 = trajectory_lqr.position_and_heading_nk3()[0, -1]
    ax.set_title('LQR Traj, End: [{:e}, {:e}, {:.03f}]'.format(*end_pose_3))
    plt.savefig('./tmp/control_pipeline_turn.png')

if __name__ == '__main__':
    visualize_control_pipeline_turn()
