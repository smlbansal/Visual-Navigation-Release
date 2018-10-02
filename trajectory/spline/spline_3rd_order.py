from trajectory.spline.spline import Spline
import tensorflow as tf


class Spline3rdOrder(Spline):
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
            f1, f2 = factors_n2[:, 0:1], factors_n2[:, 1:]
            start_pos_n12 = self.start_state.position_nk2()
            x0_n1, y0_n1 = start_pos_n12[:, :, 0], start_pos_n12[:, :, 1]
            t0_n1 = self.start_state.heading_nk1()[:, :, 0]
            v0_n1 = self.start_state.speed_nk1()[:, :, 0]
 
            goal_pos_n12 = self.goal_state.position_nk2()
            xg_n1, yg_n1 = goal_pos_n12[:, :, 0], goal_pos_n12[:, :, 1]
            tg_n1 = self.goal_state.heading_nk1()[:, :, 0]
            vg_n1 = self.goal_state.speed_nk1()[:, :, 0]

            x0, y0, t0, v0 = x0_n1, y0_n1, t0_n1, v0_n1
            xg, yg, tg, vg = xg_n1, yg_n1, tg_n1, vg_n1

            d1 = x0
            c1 = f1*tf.cos(t0)
            a1 = f2*tf.cos(tg)-2*xg+c1+2*d1
            b1 = 3*xg-f2*tf.cos(tg)-2*c1-3*d1

            d2 = y0
            c2 = f1*tf.sin(t0)
            a2 = f2*tf.sin(tg)-2*yg+c2+2*d2
            b2 = 3*yg-f2*tf.sin(tg)-2*c2-3*d2

            c3 = v0 / f1
            a3 = (vg/f2) + c3 - 2.
            b3 = 1. - c3 - a3

            self.x_coeffs = [a1, b1, c1, d1]
            self.y_coeffs = [a2, b2, c2, d2]
            self.p_coeffs = [a3, b3, c3]

    def _eval_spline(self, ts, calculate_speeds=True):
        """ Evaluates the spline on points in ts
        Assumes ts is normalized to be in [0, 1.]
        """
        a1, b1, c1, d1 = self.x_coeffs
        a2, b2, c2, d2 = self.y_coeffs
        a3, b3, c3 = self.p_coeffs

        with tf.name_scope('eval_spline'):
            t2, t3 = ts*ts, ts*ts*ts
            ps = a3*t3+b3*t2+c3*ts
            p2, p3 = ps*ps, ps*ps*ps
            xs = a1*p3+b1*p2+c1*ps+d1
            ys = a2*p3+b2*p2+d2

            ps_dot = 3*a3*t2+2*b3*ts+c3
            xs_dot = 3*a1*p2+2*b1*ps+c1
            ys_dot = 3*a2*p2+2*b2*ps+c2

            ps_ddot = 6*a3*ts+2*b3
            xs_ddot = 6*a1*ps+2*b1
            ys_ddot = 6*a2*ps+2*b2

            self._position_nk2 = tf.stack([xs, ys], axis=2)
            heading_nk = tf.atan2(ys_dot, xs_dot)
            self._heading_nk1 = heading_nk[:, :, None]

            # CHECK FOR NANS if calculating speeds!!!!
            if calculate_speeds:
                speed_ps_nk = tf.sqrt(xs_dot**2 + ys_dot**2)
                speed_nk = (speed_ps_nk*ps_dot)
                with tf.name_scope('omega'):
                    ps_sq = tf.square(ps_dot)
                    num_l = tf.multiply(ys_ddot, ps_sq) + \
                            tf.multiply(ys_dot, ps_ddot)
                    num_l = tf.multiply(num_l, tf.multiply(xs_dot, ps_dot))
                    num_r = tf.multiply(xs_ddot, ps_sq) + \
                            tf.multiply(xs_dot, ps_ddot)
                    num_r = tf.multiply(num_r, tf.multiply(ys_dot, ps_dot))
                    angular_speed_nk = (num_l + num_r) / tf.square(speed_nk)
                self._speed_ps_nk1 = speed_ps_nk[:, :, None]
                self._speed_nk1 = speed_nk[:, :, None]
                self._angular_speed_nk1 = angular_speed_nk[:, :, None]

    @staticmethod
    def check_start_goal_equivalence(start_state_old, goal_state_old, start_state_new,
                                     goal_state_new):
        """ A utility function that checks whether start_state_old,
        goal_state_old imply the same spline constraints as those implied by
        start_state_new, goal_state_new. Useful for checking that a
        precomputed spline on the old start and goal will work on new start
        and goal."""
        start_old_pos_nk2 = start_state_old.position_nk2()
        start_old_heading_nk1 = start_state_old.heading_nk1()
        start_old_speed_nk1 = start_state_old.speed_nk1()

        start_new_pos_nk2 = start_state_new.position_nk2()
        start_new_heading_nk1 = start_state_new.heading_nk1()
        start_new_speed_nk1 = start_state_new.speed_nk1()

        start_pos_match = (tf.norm(start_old_pos_nk2-start_new_pos_nk2).numpy() == 0.0)
        start_heading_match = (tf.norm(start_old_heading_nk1-start_new_heading_nk1).numpy() == 0.0)
        start_speed_match = (tf.norm(start_old_speed_nk1-start_new_speed_nk1).numpy() == 0.0)

        start_match = (start_pos_match and start_heading_match and
                       start_speed_match)

        # Check whether they are the same object
        if goal_state_old is goal_state_new:
            return start_match
        else:
            goal_old_pos_nk2 = goal_state_old.position_nk2()
            goal_old_heading_nk1 = goal_state_old.heading_nk1()
            goal_old_speed_nk1 = goal_state_old.speed_nk1()

            goal_new_pos_nk2 = goal_state_new.position_nk2()
            goal_new_heading_nk1 = goal_state_new.heading_nk1()
            goal_new_speed_nk1 = goal_state_new.speed_nk1()

            goal_pos_match = (tf.norm(goal_old_pos_nk2-goal_new_pos_nk2).numpy() == 0.0)
            goal_heading_match = (tf.norm(goal_old_heading_nk1-goal_new_heading_nk1).numpy() == 0.0)
            goal_speed_match = (tf.norm(goal_old_speed_nk1-goal_new_speed_nk1).numpy() == 0.0)

            goal_match = (goal_pos_match and goal_heading_match and
                           goal_speed_match)
            return start_match and goal_match


    def render(self, ax, batch_idx=0, freq=4):
        super().render(ax, batch_idx, freq)
        goal_n15 = self.goal_state.position_heading_speed_and_angular_speed_nk5()
        target_state = goal_n15[batch_idx, 0]
        ax.quiver([target_state[0]], [target_state[1]],
                  [tf.cos(target_state[2])],
                  [tf.sin(target_state[2])], units='width')
        ax.set_title('3rd Order Spline')
