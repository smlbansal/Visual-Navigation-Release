from .spline import Spline
import tensorflow as tf
import numpy as np

class DB_3rd_Order_Spline(Spline):

  def __init__(self):
    self.start, self.goal, self.factors = None, None, None

  def fit(self, start, goal, factors):
    self.start, self.goal, self.factors = start, goal, factors 

    with tf.name_scope('fit_spline'):
      f1, f2 = factors[0:1], factors[1:]
      xg, yg, tg = goal[0:1], goal[1:2], goal[2:3]
      v0, vf = start[3:4], goal[3:4]

      c1 = f1
      a1 = f2*tf.cos(tg)-2*xg+c1
      b1 = 3*xg-f2*tf.cos(tg)-2*c1

      a2 = f2*tf.sin(tg)-2*yg
      b2 = 3*yg-f2*tf.sin(tg)
     
      c3 = v0 / f1
      a3 = (vf/f2) + c3 - 2.
      b3 = 1. - c3 - a3 
      
      self.x_coeffs = [a1,b1,c1]
      self.y_coeffs = [a2,b2,b2*0.]
      self.p_coeffs = [a3,b3,c3]

  def evaluate(self, ts):
    a1,b1,c1 = self.x_coeffs
    a2,b2,c2 = self.y_coeffs
    a3,b3,c3 = self.p_coeffs

    with tf.name_scope('eval_spline'):
      t2, t3 = ts*ts, ts*ts*ts
      ps = a3*t3+b3*t2+c3*ts
      p2, p3 = ps*ps, ps*ps*ps
      xs = a1*p3+b1*p2+c1*ps
      ys = a2*p3+b2*p2

      ps_dot = 3*a3*t2+2*b3*ts+c3
      xs_dot = 3*a1*p2+2*b1*ps+c1
      ys_dot = 3*a2*p2+2*b2*ps
    
      speed_angle = tf.atan2(ys_dot, xs_dot)
      speed = tf.sqrt(xs_dot**2 + ys_dot**2)
      actual_speed = speed*ps_dot

      self.ps, self.xs, self.ys = ps, xs, ys
      self.ps_dot, self.xs_dot, self.ys_dot = ps_dot, xs_dot, ys_dot
      self.speed = speed

    #return a trajectory object

  def render(self, ax, freq=4):
    ps_dot, xs_dot, ys_dot = self.ps_dot, self.xs_dot, self.ys_dot
    xs, ys, speed, target_state = self.xs, self.ys, self.speed, self.goal
    ax.plot(xs, ys, 'r-')
    ax.quiver(xs[::freq], ys[::freq], xs_dot[::freq]/speed[::freq], ys_dot[::freq]/speed[::freq], units='width')
    ax.quiver([target_state[0]], [target_state[1]], [np.cos(target_state[2])], [np.sin(target_state[2])], units='width')
    ax.set_title('3rd Order Spline')

