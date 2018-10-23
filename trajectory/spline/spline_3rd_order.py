from trajectory.spline.spline import Spline
import tensorflow as tf
import numpy as np


class Spline3rdOrder(Spline):
    def __init__(self, dt, n, k, epsilon=1e-10):
        super().__init__(dt=dt, n=n, k=k)
        self.epsilon = epsilon

    """ A class representing a 3rd order spline for a mobile ground robot
    (in a 2d cartesian plane). The 3rd order spline allows for constraints
    on the start state, [x0, y0, theta0, v0], and goal state,
    [xg, yg, thetag,vg]. Angular speeds w0 and wg are not constrainable.
    """

    def fit(self, start_state, goal_state, factors_n2=None):
        """Fit a 3rd order spline between start state and goal state.
        Factors_n2 represent 2 degrees of freedom in fitting the spline.
        If factors_n2=None it is set heuristically below.
        The spline is of the form:
            p(t) = a3t^3+b3t^2+c3t+d3
            x(p) = a1p^3+b1p^2+c1p+d1
            y(p) = a2p^2+b2p^2+c2p+d2
        """

        self.start_state = start_state
        self.goal_state = goal_state

        if factors_n2 is None:  # Compute them heuristically
            factor1_n1 = self.start_state.speed_nk1()[:, :, 0] + \
                         tf.norm(goal_state.position_nk2()-start_state.position_nk2(), axis=2)
            factor2_n1 = factor1_n1
            factors_n2 = tf.concat([factor1_n1, factor2_n1], axis=1)

        with tf.name_scope('fit_spline'):
            f1_n1, f2_n1 = factors_n2[:, 0:1], factors_n2[:, 1:]

            start_pos_n12 = self.start_state.position_nk2()
            goal_pos_n12 = self.goal_state.position_nk2()

            # Multiple solutions if start and goal are the same x,y coordinates
            assert(tf.reduce_all(tf.norm(goal_pos_n12-start_pos_n12, axis=2) >= self.epsilon))

            x0_n1, y0_n1 = start_pos_n12[:, :, 0], start_pos_n12[:, :, 1]
            t0_n1 = self.start_state.heading_nk1()[:, :, 0]
            v0_n1 = self.start_state.speed_nk1()[:, :, 0]

            xg_n1, yg_n1 = goal_pos_n12[:, :, 0], goal_pos_n12[:, :, 1]
            tg_n1 = self.goal_state.heading_nk1()[:, :, 0]
            vg_n1 = self.goal_state.speed_nk1()[:, :, 0]

            d1_n1 = x0_n1
            c1_n1 = f1_n1*tf.cos(t0_n1)
            a1_n1 = f2_n1*tf.cos(tg_n1)-2*xg_n1+c1_n1+2*d1_n1
            b1_n1 = 3*xg_n1-f2_n1*tf.cos(tg_n1)-2*c1_n1-3*d1_n1

            d2_n1 = y0_n1
            c2_n1 = f1_n1*tf.sin(t0_n1)
            a2_n1 = f2_n1*tf.sin(tg_n1)-2*yg_n1+c2_n1+2*d2_n1
            b2_n1 = 3*yg_n1-f2_n1*tf.sin(tg_n1)-2*c2_n1-3*d2_n1

            c3_n1 = v0_n1 / f1_n1
            a3_n1 = (vg_n1/f2_n1) + c3_n1 - 2.
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

            if calculate_speeds:
                ts_dot_n4k = tf.stack([3.0*tf.pow(ts_nk, 2), 2.0*ts_nk,
                                       tf.ones_like(ts_nk), tf.zeros_like(ts_nk)],
                                      axis=1)
                ps_ddot_n4k = tf.stack([6.0*ps_nk, 2.0*tf.ones_like(ps_nk),
                                        tf.zeros_like(ps_nk),
                                        tf.zeros_like(ps_nk)], axis=1)

                ps_dot_nk = tf.squeeze(tf.matmul(p_coeffs_n14, ts_dot_n4k), axis=1)

                xs_ddot_nk = tf.squeeze(tf.matmul(x_coeffs_n14, ps_ddot_n4k), axis=1)
                ys_ddot_nk = tf.squeeze(tf.matmul(y_coeffs_n14, ps_ddot_n4k), axis=1)

                speed_ps_nk = tf.sqrt(xs_dot_nk**2 + ys_dot_nk**2)
                speed_nk = (speed_ps_nk*ps_dot_nk)

                numerator_nk = xs_dot_nk*ys_ddot_nk-ys_dot_nk*xs_ddot_nk
                angular_speed_nk = numerator_nk/(speed_ps_nk**2) * ps_dot_nk

                self._speed_nk1 = speed_nk[:, :, None]
                self._angular_speed_nk1 = angular_speed_nk[:, :, None]

    def check_dynamic_feasability(self, speed_max_system, angular_speed_max_system, horizon_s):
        """Checks whether the current computed spline (on time points in [0, 1])
        can be executed in time <= horizon_s (specified in seconds) while respecting max speed and
        angular speed constraints. Returns the batch indices of all valid splines."""
        # Speed assumed to be in [0, speed_max_system]
        # Angular speed assumed to be in [-angular_speed_max_system, angular_speed_max_system]
        max_speed = tf.reduce_max(self.speed_nk1(), axis=1)
        max_angular_speed = tf.reduce_max(tf.abs(self.angular_speed_nk1()), axis=1)

        horizon_opt_speed = max_speed/speed_max_system
        horizon_opt_angular_speed = max_angular_speed/angular_speed_max_system
        horizons = tf.concat([horizon_opt_speed, horizon_opt_angular_speed], axis=1)
        cutoff_horizon = tf.reduce_max(horizons, axis=1)
        valid_idxs = tf.squeeze(tf.where(cutoff_horizon <= horizon_s), axis=1)
        return tf.cast(valid_idxs, tf.int32)

    @staticmethod
    def check_start_goal_equivalence(start_state_old, goal_state_old,
                                     start_state_new, goal_state_new):
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

    @staticmethod
    def ensure_goals_valid(start_x, start_y, goal_x_nk1, goal_y_nk1, goal_theta_nk1, epsilon):
        """ Perturbs goal_x and goal_y by epsilon if needed ensuring that a unique spline exists.
        Assumes that all goal angles are within [-pi/2., pi/2]."""
        assert((goal_theta_nk1 >= -np.pi/2.).all() and (goal_theta_nk1 <= np.pi/2.).all())
        norms = np.linalg.norm(np.concatenate([goal_x_nk1-start_x, goal_y_nk1-start_y], axis=2),
                               axis=2)
        invalid_idxs = (norms == 0.0)
        goal_x_nk1[invalid_idxs] += epsilon
        goal_y_nk1[invalid_idxs] += np.sign(np.sin(goal_theta_nk1[invalid_idxs]))*epsilon
        return goal_x_nk1, goal_y_nk1, goal_theta_nk1

    def render(self, axs, batch_idx=0, freq=4, plot_control=False):
        """Render the spline trajectory from batch_idx
        including goal position."""
        super().render(axs, batch_idx, freq, plot_control=plot_control)
        goal_n15 = self.goal_state.position_heading_speed_and_angular_speed_nk5()
        target_state = goal_n15[batch_idx, 0]
        ax = axs[0]
        ax.quiver([target_state[0]], [target_state[1]],
                  [tf.cos(target_state[2])],
                  [tf.sin(target_state[2])], units='width')
        ax.set_title('Spline Trajectory')
