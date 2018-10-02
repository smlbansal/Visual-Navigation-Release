import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from trajectory.trajectory import State
tf.enable_eager_execution()


def test_spline_3rd_order(visualize=False):
    np.random.seed(seed=1)
    n = 5
    dt = .01
    k = 100

    target_state = np.random.uniform(-np.pi, np.pi, 3)
    v0 = np.random.uniform(0., 0.5, 1)[0]  # Initial speed
    vf = 0.

    # Initial State is [0, 0, 0, v0, 0]
    start_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32)*v0

    goal_posx_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[0]
    goal_posy_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[1]
    goal_pos_nk2 = tf.concat([goal_posx_nk1, goal_posy_nk1], axis=2)
    goal_heading_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[2]
    goal_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * vf

    start_state = State(dt, n, 1, speed_nk1=start_speed_nk1, variable=False)
    goal_state = State(dt, n, 1, position_nk2=goal_pos_nk2,
                       speed_nk1=goal_speed_nk1, heading_nk1=goal_heading_nk1,
                       variable=True)

    start_nk5 = start_state.position_heading_speed_and_angular_speed_nk5()
    start_n5 = start_nk5[:, 0]

    goal_nk5 = goal_state.position_heading_speed_and_angular_speed_nk5()
    goal_n5 = goal_nk5[:, 0]

    ts_nk = tf.tile(tf.linspace(0., dt*k, k)[None], [n, 1])
    spline_traj = Spline3rdOrder(dt=dt, k=k, n=n)
    spline_traj.fit(start_state, goal_state, factors_n2=None)
    spline_traj.eval_spline(ts_nk, calculate_speeds=True)

    pos_nk3 = spline_traj.position_and_heading_nk3()
    v_nk1 = spline_traj.speed_nk1()
    start_pos_diff = (pos_nk3 - start_n5[:, None, :3])[:, 0]
    goal_pos_diff = (pos_nk3 - goal_n5[:, None, :3])[:, -1]
    assert(np.allclose(start_pos_diff, np.zeros((n, 3)), atol=1e-6))
    assert(np.allclose(goal_pos_diff, np.zeros((n, 3)), atol=1e-6))

    start_vel_diff = (v_nk1 - start_n5[:, None, 3:4])[:, 0]
    goal_vel_diff = (v_nk1 - goal_n5[:, None, 3:4])[:, -1]
    assert(np.allclose(start_vel_diff, np.zeros((n, 1)), atol=1e-6))
    assert(np.allclose(goal_vel_diff, np.zeros((n, 1)), atol=1e-6))

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        spline_traj.render(ax, freq=4)
        plt.show()


if __name__ == '__main__':
    test_spline_3rd_order(visualize=True)
