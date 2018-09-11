import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import pdb

def test_db_3rd_order():
  from control.spline.db_3rd_order_spline import DB3rdOrderSpline
  np.random.seed(seed=1)

  target_state = np.random.uniform(-np.pi, np.pi, 3)
  v0 = np.random.uniform(0., 0.5, 1)[0] # Initial speed
  vf = 0.

  #[x,y,theta,v,omega]
  start = [0., 0., 0., v0, 0.]
  goal = [target_state[0], target_state[1], target_state[2], 0., 0.]
  factor1 = np.linalg.norm(target_state[:2])
  factor2 = np.linalg.norm(target_state[:2])

  time_samples = 100
  ts = np.linspace(0., 1., time_samples)

  start_t = tf.convert_to_tensor(start, name='start', dtype=tf.float32)
  goal_t = tf.convert_to_tensor(goal, name='goal', dtype=tf.float32)
  factors_t = tf.convert_to_tensor([factor1, factor2], name='factors', dtype=tf.float32)
  ts_t = tf.convert_to_tensor(ts, name='ts', dtype=tf.float32)
  
  db_spline = DB3rdOrderSpline()
  db_spline.fit(start_t, goal_t, factors_t)
  db_spline.evaluate(ts_t)
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  db_spline.render(ax)
  plt.show()

if __name__ == '__main__':
  test_db_3rd_order()

