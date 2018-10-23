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
from control_pipelines.control_pipeline import Control_Pipeline_v0
from utils import utils
from dotmap import DotMap


def create_params():
    p = DotMap()
    p.seed = 1
    p.planning_horizon_s = 3.0#1.5  # seconds
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
    p.spline_params = {}
    p.control_pipeline_params = {}

    p._cost = QuadraticRegulatorRef
    p._spline = Spline3rdOrder
    p._control_pipeline = Control_Pipeline_v0
    return p


def visualize_control_pipeline_turn(v0, theta_goal, axes):
    epsilon = 1e-5
    p = create_params()
    np.random.seed(seed=p.seed)
    tf.set_random_seed(seed=p.seed)
    n = p.n
    dt = p.dx

    # Initial State is [0, 0, 0, v0, 0]
    x0, y0 = .0, 0.
    start_pos_nk2 = np.array([[[x0, y0]]], dtype=np.float32)
    start_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32)*v0

    vf = 0.
    xg, yg = 0.5, 0.5
    goal_x = np.ones((n,1,1), dtype=np.float32)*xg
    goal_y = np.ones((n, 1, 1), dtype=np.float32)*yg
    goal_theta = np.ones((n, 1, 1), dtype=np.float32) * theta_goal
    goal_x, goal_y, goal_theta = p._spline.ensure_goals_valid(x0, y0, goal_x, goal_y, goal_theta,
                                                              epsilon=epsilon)

    goal_pos_nk2 = np.concatenate([goal_x, goal_y], axis=2)
    goal_heading_nk1 = goal_theta
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
    
    wpt_13 = waypt_n5[0, :3]
    ax = axes[0]
    traj_spline.render(ax, batch_idx=0, freq=1)
    ax.set_title('Spline, Wpt: [{:e}, {:e}, {:.03f}]'.format(wpt_13[0],
                                                                  wpt_13[1],
                                                                  wpt_13[2]))

    ax = axes[1]
    spline_heading = traj_spline.heading_nk1()[0, :, 0]
    ax.plot(spline_heading.numpy(), 'r--')
    ax.set_title('Spline Theta')

    ax = axes[2]
    trajectory_lqr.render(ax, batch_idx=0, freq=1)
    end_pose_3 = trajectory_lqr.position_and_heading_nk3()[0, -1]
    ax.set_title('LQR Traj, End: [{:e}, {:e}, {:.03f}]'.format(*end_pose_3))
   
    ax = axes[3]
    omega = trajectory_lqr.angular_speed_nk1()[0, :, 0]
    ax.plot(omega.numpy(), 'r--')
    ax.set_title('Omega')

    
    ax = axes[4]
    velocity = trajectory_lqr.speed_nk1()[0, :, 0]
    ax.plot(velocity.numpy(), 'r--')
    ax.set_title('Velocity')




if __name__ == '__main__':
    plt.style.use('ggplot')
    N = 21
    #thetas = np.linspace(-np.pi/2., np.pi/2., N)
    thetas = [np.pi/3.]
    N = len(thetas)
    fig, _, axess = utils.subplot2(plt, (N, 5), (8, 8), (.4, .4))
    v0 = 0.1
    for theta_goal in thetas:
        axes = [axess.pop(), axess.pop(), axess.pop(), axess.pop(), axess.pop()]
        visualize_control_pipeline_turn(v0, theta_goal, axes)
    fig.suptitle('Turning Trajectories for v0={:.3f}, vf={:.3f}'.format(v0, 0.0))
    plt.savefig('./tmp/control_pipeline_turn_v0_{:.2f}.png'.format(v0))
