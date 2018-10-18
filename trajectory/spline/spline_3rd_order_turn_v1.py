from trajectory.spline.spline_3rd_order import Spline3rdOrder
import tensorflow as tf


class Spline3rdOrderTurnV1(Spline3rdOrder):
    def __init__(self, dt, n, k, epsilon=1e-10):
        super().__init__(dt=dt, n=n, k=k)
        self.epsilon = epsilon

    """ A class representing a 3rd order spline for a mobile ground robot
    (in a 2d cartesian plane). The 3rd order spline allows for constraints
    on the start state, [x0, y0, theta0, v0], and goal state,
    [xg, yg, thetag,vg]. Angular speeds w0 and wg are not constrainable.
    """

    def fit(self, start_state, goal_state, factors_n2=None):
        self.start_state = start_state
        self.goal_state = goal_state
        # compute them heuristically based on dist to goal
        if factors_n2 is None:
            factors_n1 = tf.norm(goal_state.position_nk2(), axis=2)
            factors_n2 = tf.concat([factors_n1, factors_n1], axis=1)
        
        with tf.name_scope('fit_spline'):
            f1_n1, f2_n1 = factors_n2[:, 0:1], factors_n2[:, 1:]
 
            start_pos_n12 = self.start_state.position_nk2()
            goal_pos_n12 = self.goal_state.position_nk2()
            
            # Multiple solutions if start and goal are the same x,y coordinates
            assert(tf.reduce_all(tf.norm(goal_pos_n12-start_pos_n12, axis=2) > self.epsilon))
 
            x0_n1, y0_n1 = start_pos_n12[:, :, 0], start_pos_n12[:, :, 1]
            t0_n1 = self.start_state.heading_nk1()[:, :, 0]
            v0_n1 = self.start_state.speed_nk1()[:, :, 0]

            xg_n1, yg_n1 = goal_pos_n12[:, :, 0], goal_pos_n12[:, :, 1]
            tg_n1 = self.goal_state.heading_nk1()[:, :, 0]
            vg_n1 = self.goal_state.speed_nk1()[:, :, 0]

            a2_n1 = yg_n1
            b2_n1 = 0.*a2_n1
            c2_n1 = 0.*a2_n1
            d2_n1 = 0.*a2_n1

            f2_n1 = 2*a2_n1/tf.sin(tg_n1)

            c1_n1 = 0.*a2_n1
            a1_n1 = f2_n1 * tf.cos(tg_n1)-xg_n1
            b1_n1 = xg_n1 - a1_n1
            d1_n1 = 0.*a2_n1

            c3_n1 = v0_n1
            a3_n1 = vg_n1 + c3_n1 - 2.
            b3_n1 = 1. - c3_n1 - a3_n1

            self.x_coeffs_n14 = tf.stack([a1_n1, b1_n1, c1_n1, d1_n1], axis=2)
            self.y_coeffs_n14 = tf.stack([a2_n1, b2_n1, c2_n1, d2_n1], axis=2)
            self.p_coeffs_n14 = tf.stack([a3_n1, b3_n1, c3_n1, 0.0*c3_n1],
                                         axis=2)

    def _eval_spline(self, ts_nk, calculate_speeds=True):
        """ Evaluates the spline on points in ts_nk
        Assumes ts is normalized to be in [0, 1.]
        """
        x_coeffs_n14 = self.x_coeffs_n14
        y_coeffs_n14 = self.y_coeffs_n14
        p_coeffs_n14 = self.p_coeffs_n14

        with tf.name_scope('eval_spline'):
            ts_n4k = tf.stack([tf.pow(ts_nk, 3), tf.pow(ts_nk, 2),
                               ts_nk, tf.ones_like(ts_nk)], axis=1)
            ps_nk = tf.squeeze(tf.matmul(p_coeffs_n14, ts_n4k), axis=1)

            ps_n4k = tf.stack([tf.pow(ps_nk, 3), tf.pow(ps_nk, 2),
                               ps_nk, tf.ones_like(ps_nk)], axis=1)
            ps_dot_n4k = tf.stack([3.0*tf.pow(ps_nk, 2), 2.0*ps_nk,
                                   tf.ones_like(ps_nk), tf.zeros_like(ps_nk)],
                                  axis=1)

            xs_nk = tf.squeeze(tf.matmul(x_coeffs_n14, ps_n4k), axis=1)
            ys_nk = tf.squeeze(tf.matmul(y_coeffs_n14, ps_n4k), axis=1)

            xs_dot_nk = tf.squeeze(tf.matmul(x_coeffs_n14, ps_dot_n4k), axis=1)
            ys_dot_nk = tf.squeeze(tf.matmul(y_coeffs_n14, ps_dot_n4k), axis=1)

            self._position_nk2 = tf.stack([xs_nk, ys_nk], axis=2)
            self._heading_nk1 = tf.atan2(ys_dot_nk, xs_dot_nk)[:, :, None]

            # CHECK FOR NANS if calculating speeds!!!!
            if calculate_speeds:
                ts_dot_n4k = tf.stack([3.0*tf.pow(ts_nk, 2), 2.0*ts_nk,
                                       tf.ones_like(ts_nk), tf.zeros_like(ts_nk)],
                                      axis=1)
                ts_ddot_n4k = tf.stack([6.0*ts_nk, 2.0*tf.ones_like(ts_nk),
                                        tf.zeros_like(ts_nk),
                                        tf.zeros_like(ts_nk)], axis=1)
                ps_ddot_n4k = tf.stack([6.0*ps_nk, 2.0*tf.ones_like(ps_nk),
                                        tf.zeros_like(ps_nk),
                                        tf.zeros_like(ps_nk)], axis=1)

                ps_dot_nk = tf.squeeze(tf.matmul(p_coeffs_n14, ts_dot_n4k), axis=1)

                ps_ddot_nk = tf.squeeze(tf.matmul(p_coeffs_n14, ts_ddot_n4k), axis=1)
                xs_ddot_nk = tf.squeeze(tf.matmul(x_coeffs_n14, ps_ddot_n4k), axis=1)
                ys_ddot_nk = tf.squeeze(tf.matmul(y_coeffs_n14, ps_ddot_n4k), axis=1)

                speed_ps_nk = tf.sqrt(xs_dot_nk**2 + ys_dot_nk**2)
                speed_nk = (speed_ps_nk*ps_dot_nk)
                with tf.name_scope('omega'):
                    ps_sq_nk = tf.square(ps_dot_nk)
                    num_l_nk = ys_ddot_nk*ps_sq_nk + ys_dot_nk*ps_ddot_nk
                    num_l_nk = num_l_nk*xs_dot_nk*ps_dot_nk
                    num_r_nk = xs_ddot_nk*ps_sq_nk + xs_dot_nk*ps_ddot_nk
                    num_r_nk = num_r_nk*ys_dot_nk*ps_dot_nk
                    angular_speed_nk = (num_l_nk + num_r_nk) / tf.square(speed_nk)
                self._speed_ps_nk1 = speed_ps_nk[:, :, None]
                self._speed_nk1 = speed_nk[:, :, None]
                self._angular_speed_nk1 = angular_speed_nk[:, :, None]
