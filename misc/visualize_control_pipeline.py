import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from systems.dubins_v3 import Dubins_v3
from systems.dubins_v2 import Dubins_v2
from systems.dubins_v1 import Dubins_v1
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from trajectory.trajectory import State
from control_pipelines.control_pipeline_v0 import Control_Pipeline_v0
from control_pipelines.control_pipeline_v1 import Control_Pipeline_v1
from utils import utils
from dotmap import DotMap


def create_params():
    p = DotMap()
    p.seed = 1
    p.planning_horizon_s = 3.0  # seconds
    p.n = 1
    p.map_bounds = [[-2.0, -2.0], [2.0, 2.0]]
    p.dx, p.dt = .05, .1
    p.k = int(np.ceil(p.planning_horizon_s/p.dt))

    p._plant = Dubins_v2
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
    p.spline_params = {'epsilon': 1e-10}
    p.control_pipeline_params = {}

    p._cost = QuadraticRegulatorRef
    p._spline = Spline3rdOrder
    p._control_pipeline = Control_Pipeline_v1
    return p


def visualize_control_pipeline(starts_n5, goals_n5):
    p = create_params()
    np.random.seed(seed=p.seed)
    tf.set_random_seed(seed=p.seed)
    p.n = len(starts_n5)
    n = p.n
    dt = p.dx

    plant = p._plant(**p.plant_params)
    control_pipeline = p._control_pipeline(system_dynamics=plant,
                                           params=p,
                                           **p.control_pipeline_params)

    start_x_n11, start_y_n11 = starts_n5[:, 0:1, None], starts_n5[:, 1:2, None]
    goal_x_n11, goal_y_n11 = goals_n5[:, 0:1, None], goals_n5[:, 1:2, None]
    goal_theta_n11 = goals_n5[:, 2:3, None]

    goal_x_n11, goal_y_n11, goal_theta_n11 = p._spline.ensure_goals_valid(start_x_n11, start_y_n11, goal_x_n11,
                                                                          goal_y_n11, goal_theta_n11,
                                                                          epsilon=control_pipeline.traj_spline.epsilon)

    start_pos_nk2 = np.concatenate([start_x_n11, start_y_n11], axis=2)
    goal_pos_nk2 = np.concatenate([goal_x_n11, goal_y_n11], axis=2)

    start_state = State(dt, n, 1, position_nk2=start_pos_nk2, heading_nk1=starts_n5[:, 2:3, None],
                        speed_nk1=starts_n5[:, 3:4, None], variable=False)
    goal_state = State(dt, n, 1, position_nk2=goal_pos_nk2,
                       speed_nk1=goals_n5[:, 3:4, None], heading_nk1=goal_theta_n11,
                       variable=True)

    trajectory_lqr = control_pipeline.plan(start_state=start_state,
                                           goal_state=goal_state)

    traj_spline = control_pipeline.traj_spline

    fig, _, axes = utils.subplot2(plt, (p.n, 6), (8, 8), (.4, .4))
    axes = axes[::-1]
    for i in range(p.n):
        axs = axes[6*i:6*(i+1)]
        traj_spline.render(axs[:3], batch_idx=i, freq=4, plot_control=True)
        trajectory_lqr.render(axs[3:], batch_idx=i, freq=4, plot_control=True,
                              label_start_and_end=True, name='LQR')
    fig.suptitle('Control Pipeline Trajectories')
    plt.savefig('./tmp/control_pipeline.png')


if __name__ == '__main__':
    plt.style.use('ggplot')

    # [x, y, theta, v, omega]
    starts_n5 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    goals_n5 = np.array([[0.5, -.8, -np.pi/2., 0., 0.]], dtype=np.float32)
    assert(len(starts_n5) == len(goals_n5))

    visualize_control_pipeline(starts_n5, goals_n5)
