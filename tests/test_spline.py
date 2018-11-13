import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from trajectory.trajectory import SystemConfig
from dotmap import DotMap
tf.enable_eager_execution()


def test_spline_3rd_order(visualize=False):
    np.random.seed(seed=1)
    n = 5
    dt = .01
    k = 100

    target_state = np.random.uniform(-np.pi, np.pi, 3)
    v0 = np.random.uniform(0., 0.5, 1)[0]  # Initial speed
    vf = 0.

    # Initial SystemConfig is [0, 0, 0, v0, 0]
    start_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32)*v0

    goal_posx_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[0]
    goal_posy_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[1]
    goal_pos_nk2 = tf.concat([goal_posx_nk1, goal_posy_nk1], axis=2)
    goal_heading_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[2]
    goal_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * vf

    start_config = SystemConfig(dt, n, 1, speed_nk1=start_speed_nk1, variable=False)
    goal_config = SystemConfig(dt, n, 1, position_nk2=goal_pos_nk2,
                               speed_nk1=goal_speed_nk1, heading_nk1=goal_heading_nk1,
                               variable=True)

    start_nk5 = start_config.position_heading_speed_and_angular_speed_nk5()
    start_n5 = start_nk5[:, 0]

    goal_nk5 = goal_config.position_heading_speed_and_angular_speed_nk5()
    goal_n5 = goal_nk5[:, 0]

    p = DotMap(spline_params=DotMap(epsilon=1e-5))
    ts_nk = tf.tile(tf.linspace(0., dt*k, k)[None], [n, 1])
    spline_traj = Spline3rdOrder(dt=dt, k=k, n=n, params=p.spline_params)
    spline_traj.fit(start_config, goal_config, factors=None)
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


def test_spline_rescaling():
    # Set the random seed
    np.random.seed(seed=1)
    
    # Spline trajectory params
    n = 2
    dt = .1
    k = 10
    final_times_n1 = tf.constant([[2.], [1.]])
    
    # Goal states and initial speeds
    goal_posx_n11 = tf.constant([[[0.4]], [[1.]]])
    goal_posy_n11 = tf.constant([[[0.]], [[1.]]])
    goal_heading_n11 = tf.constant([[[0.]], [[np.pi/2]]])
    start_speed_nk1 = tf.ones((2, 1, 1), dtype=tf.float32) * 0.5
    
    # Define the maximum speed, angular speed and maximum horizon
    max_speed = 0.6
    max_angular_speed = 1.1
    
    # Define start and goal configurations
    start_config = SystemConfig(dt, n, 1, speed_nk1=start_speed_nk1, variable=False)
    goal_config = SystemConfig(dt, n,
                               k=1,
                               position_nk2=tf.concat([goal_posx_n11, goal_posy_n11], axis=2),
                               heading_nk1=goal_heading_n11,
                               variable=True)
    
    # Fit the splines
    p = DotMap(spline_params=DotMap(epsilon=1e-5))
    spline_trajs = Spline3rdOrder(dt=dt, k=k, n=n, params=p.spline_params)
    spline_trajs.fit(start_config, goal_config, final_times_n1, factors=None)
    
    # Evaluate the splines
    ts_nk = tf.stack([tf.linspace(0., final_times_n1[0, 0], 100),
                      tf.linspace(0., final_times_n1[1, 0], 100)], axis=0)
    spline_trajs.eval_spline(ts_nk, calculate_speeds=True)
    
    # Compute the required horizon
    required_horizon_n1 = spline_trajs.compute_dynamically_feasible_horizon(max_speed, max_angular_speed)
    assert required_horizon_n1[0, 0] < final_times_n1[0, 0]
    assert required_horizon_n1[1, 0] > final_times_n1[1, 0]
    
    # Compute the maximum speed and angular speed
    max_speed_n1 = tf.reduce_max(spline_trajs.speed_nk1(), axis=1)
    max_angular_speed_n1 = tf.reduce_max(tf.abs(spline_trajs.angular_speed_nk1()), axis=1)
    assert max_speed_n1[0, 0] < max_speed
    assert max_angular_speed_n1[0, 0] < max_angular_speed
    assert max_speed_n1[1, 0] > max_speed
    assert max_angular_speed_n1[1, 0] > max_angular_speed
    
    # Rescale horizon so that the trajectories are dynamically feasible
    spline_trajs.rescale_spline_horizon_to_dynamically_feasible_horizon(max_speed, max_angular_speed)
    assert np.allclose(spline_trajs.final_times_n1.numpy(), required_horizon_n1.numpy(), atol=1e-2)
    
    # Compute the maximum speed and angular speed
    max_speed_n1 = tf.reduce_max(spline_trajs.speed_nk1(), axis=1)
    max_angular_speed_n1 = tf.reduce_max(tf.abs(spline_trajs.angular_speed_nk1()), axis=1)
    assert max_speed_n1[0, 0] <= max_speed
    assert max_angular_speed_n1[0, 0] <= max_angular_speed
    assert max_speed_n1[1, 0] <= max_speed
    assert max_angular_speed_n1[1, 0] <= max_angular_speed
    
    # Find the spline trajectories that are valid
    valid_idxs_n = spline_trajs.find_trajectories_within_a_horizon(horizon_s=2.)
    assert valid_idxs_n.shape == (1,)
    assert valid_idxs_n.numpy()[0] == 0
    

if __name__ == '__main__':
    test_spline_3rd_order(visualize=False)
    test_spline_rescaling()
