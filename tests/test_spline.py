import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

def test_db_3rd_order():
    from trajectory.spline.db_3rd_order_spline import DB3rdOrderSpline
    np.random.seed(seed=1)
    n=5
    dt = .01
    k = 100

    target_state = np.random.uniform(-np.pi, np.pi, 3)
    v0 = np.random.uniform(0., 0.5, 1)[0] # Initial speed
    vf = 0.

    start = [0., 0., 0., v0, 0.]
    goal = [target_state[0], target_state[1], target_state[2], 0., 0.]

    start_n5 = np.tile(start, n).reshape((n,5))
    goal_n5 = np.tile(goal, n).reshape((n,5))
    
    start_n5 = tf.constant(start_n5, name='start', dtype=tf.float32)
    goal_n5 = tfe.Variable(goal_n5, name='goal', dtype=tf.float32)

    db_spline_traj = DB3rdOrderSpline(dt=dt, k=k, n=n, start_n5=start_n5)
    db_spline_traj.fit(goal_n5=goal_n5, factors_n2=None)
    db_spline_traj.evaluate()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    db_spline_traj.render(ax, freq=4)
    plt.show()

if __name__ == '__main__':
    test_db_3rd_order()

