from trajectory.spline.spline  import Spline
import tensorflow as tf

class DB3rdOrderSpline(Spline):
    def __init__(self, dt, k, start_n5, goal_n5, factors_n2):
        super().__init__(dt=dt, k=k)
        self.ts = tf.linspace(0., dt*k, k)
        self.start_n5, self.goal_n5, self.factors_n2 = start_n5, goal_n5, factors_n2
        self.fit()
        self.evaluate()
 
    def fit(self):
        start_n5, goal_n5, factors_n2 = self.start_n5, self.goal_n5, self.factors_n2
        with tf.name_scope('fit_spline'):
            f1, f2 = factors_n2[:,0:1], factors_n2[:,1:]
            xg, yg, tg = goal_n5[:,0:1], goal_n5[:,1:2], goal_n5[:,2:3]
            v0, vf = start_n5[:,3:4], goal_n5[:,3:4]

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

    def evaluate(self):
        ts = self.ts
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

            ps_ddot = 6*a3*ts+2*b3
            xs_ddot = 6*a1*ps+2*b1
            ys_ddot = 6*a2*ps+2*b2

            self._position_nk2 = tf.stack([xs,ys],axis=2)
            self._speed_ps_nk1 = tf.sqrt(xs_dot**2 + ys_dot**2)
            self._speed_nk1 = self._speed_ps_nk1*ps_dot
            self._heading_nk1 = tf.atan2(ys_dot, xs_dot)
            
            with tf.name_scope('omega'):
                ps_sq = tf.square(ps_dot)
                num_l = tf.multiply(ys_ddot, ps_sq) + tf.multiply(ys_dot, ps_ddot)
                num_l = tf.multiply(num_l, tf.multiply(xs_dot, ps_dot))
                num_r = tf.multiply(xs_ddot, ps_sq) + tf.multiply(xs_dot, ps_ddot)
                num_r = tf.multiply(num_r, tf.multiply(ys_dot, ps_dot))
                self._angular_speed_nk1 = (num_l + num_r) / tf.square(self._speed_nk1)

    def render(self, ax, freq=4):
        if self._heading_nk1.shape[0].value > 1:
            print('Warning. Splines generated for multiple trajectories. Only the 1st will be rendered')
        
        xs, ys, thetas = self._position_nk2[0,:,0], self._position_nk2[0,:,1], self._heading_nk1[0]
        speed = self._speed_ps_nk1[0]
        target_state = self.goal_n5[0]
        xs_dot, ys_dot = tf.cos(thetas), tf.sin(thetas)
        ax.plot(xs, ys, 'r-')
        ax.quiver(xs[::freq], ys[::freq], xs_dot[::freq]/speed[::freq], ys_dot[::freq]/speed[::freq], units='width')
        ax.quiver([target_state[0]], [target_state[1]], [tf.cos(target_state[2])], [tf.sin(target_state[2])], units='width')
        ax.set_title('3rd Order Spline')

