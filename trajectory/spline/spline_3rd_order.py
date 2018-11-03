from trajectory.spline.spline import Spline
import tensorflow as tf
import numpy as np


class Spline3rdOrder(Spline):
    def __init__(self, dt, n, k, params):
        super().__init__(dt=dt, n=n, k=k)
        self.params = params

    """ A class representing a 3rd order spline for a mobile ground robot
    (in a 2d cartesian plane). The 3rd order spline allows for constraints
    on the start config, [x0, y0, theta0, v0], and goal config,
    [xg, yg, thetag,vg]. Angular speeds w0 and wg are not constrainable.
    """

    def fit(self, start_config, goal_config, final_times_n1=None, factors=None):
        """Fit a 3rd order spline between start config and goal config.
        Factors_n2 represent 2 degrees of freedom in fitting the spline.
        If factors_n2=None it is set heuristically below.
        If final_time_n1=None, a final time of 1 is used.
        The spline is of the form:
            p(t) = a3(t/tf)^3+b3(t/tf)^2+c3(t/tf)+d3
            x(p) = a1p^3+b1p^2+c1p+d1
            y(p) = a2p^2+b2p^2+c2p+d2
        """

        # Compute the factors
        if factors is None:  # Compute them heuristically
            factor1_n1 = start_config.speed_nk1()[:, :, 0] + \
                         tf.norm(goal_config.position_nk2()-start_config.position_nk2(), axis=2)
            factor2_n1 = factor1_n1
            factors_n2 = tf.concat([factor1_n1, factor2_n1], axis=1)
        else:
            factors_n2 = factors

        # Compute the final times
        if final_times_n1 is None:
            final_times_n1 = tf.ones((self.n, 1))

        # Fit spline
        with tf.name_scope('fit_spline'):
            f1_n1, f2_n1 = factors_n2[:, 0:1], factors_n2[:, 1:]

            start_pos_n12 = start_config.position_nk2()
            goal_pos_n12 = goal_config.position_nk2()

            # Multiple solutions if start and goal are the same x,y coordinates
            assert(tf.reduce_all(tf.norm(goal_pos_n12-start_pos_n12, axis=2) >=
                                 self.params.epsilon))

            x0_n1, y0_n1 = start_pos_n12[:, :, 0], start_pos_n12[:, :, 1]
            t0_n1 = start_config.heading_nk1()[:, :, 0]
            v0_n1 = start_config.speed_nk1()[:, :, 0]

            xg_n1, yg_n1 = goal_pos_n12[:, :, 0], goal_pos_n12[:, :, 1]
            tg_n1 = goal_config.heading_nk1()[:, :, 0]
            vg_n1 = goal_config.speed_nk1()[:, :, 0]

            d1_n1 = x0_n1
            c1_n1 = f1_n1*tf.cos(t0_n1)
            a1_n1 = f2_n1*tf.cos(tg_n1)-2*xg_n1+c1_n1+2*d1_n1
            b1_n1 = 3*xg_n1-f2_n1*tf.cos(tg_n1)-2*c1_n1-3*d1_n1

            d2_n1 = y0_n1
            c2_n1 = f1_n1*tf.sin(t0_n1)
            a2_n1 = f2_n1*tf.sin(tg_n1)-2*yg_n1+c2_n1+2*d2_n1
            b2_n1 = 3*yg_n1-f2_n1*tf.sin(tg_n1)-2*c2_n1-3*d2_n1

            c3_n1 = final_times_n1 * v0_n1 / f1_n1
            a3_n1 = (final_times_n1*vg_n1/f2_n1) + c3_n1 - 2.
            b3_n1 = 1. - c3_n1 - a3_n1

            self.x_coeffs_n14 = tf.stack([a1_n1, b1_n1, c1_n1, d1_n1], axis=2)
            self.y_coeffs_n14 = tf.stack([a2_n1, b2_n1, c2_n1, d2_n1], axis=2)
            self.p_coeffs_n14 = tf.stack([a3_n1, b3_n1, c3_n1, 0.0*c3_n1],
                                         axis=2)
            self.final_times_n1 = final_times_n1

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

    def free_memory(self):
        """Assumes that a spline has already been fit and evaluated and
        that the user will not need to fit or evaluate it again (as is the case
        when precomputing splines in egocentric coordinates). Set's irrelevant
        instance variables to None to be garbage collected. Note: won't do anything
        with a static tensorflow graph."""
        self.x_coeffs_n14 = None
        self.y_coeffs_n14 = None
        self.p_coeffs_n14 = None

    # TODO: Probably Delete this Later. It is now redundant
    def enforce_dynamic_feasability(self, system_dynamics, horizon_s):
        """Checks whether the current computed spline (on time points in [0, horizon_s])
        can be executed in time <= horizon_s (specified in seconds) while respecting max speed and
        angular speed constraints. Returns the batch indices of all valid splines."""
        if system_dynamics.v_bounds is None and system_dynamics.w_bounds is None:
            return self.n

        # Speed assumed to be in [0, speed_max_system]
        # Angular speed assumed to be in [-angular_speed_max_system, angular_speed_max_system]
        speed_max_system = system_dynamics.v_bounds[1]
        angular_speed_max_system = system_dynamics.w_bounds[1]

        max_speed = tf.reduce_max(self.speed_nk1()*horizon_s, axis=1)
        max_angular_speed = tf.reduce_max(tf.abs(self.angular_speed_nk1()*horizon_s), axis=1)

        horizon_speed = max_speed/speed_max_system
        horizon_angular_speed = max_angular_speed/angular_speed_max_system
        horizons = tf.concat([horizon_speed, horizon_angular_speed], axis=1)
        cutoff_horizon_n = tf.reduce_max(horizons, axis=1)
        valid_idxs = tf.squeeze(tf.where(cutoff_horizon_n <= horizon_s), axis=1)

        # If vary_horizon is true the spline is dynamically recomputed with
        # time horizon min(cutoff_horizon_n, horizon_s) along the batch dimension
        if self.params.spline_params.vary_horizon:
            ts_nk = tf.tile(tf.linspace(0., horizon_s, self.k)[None], [self.n, 1])
            valid_mask_nk = (ts_nk <= cutoff_horizon_n[:, None])
            valid_mask_inv_nk = tf.cast(tf.logical_not(valid_mask_nk), dtype=tf.float32)
            valid_mask_nk = tf.cast(valid_mask_nk, dtype=tf.float32)

            ts_nk = ts_nk*valid_mask_nk + valid_mask_inv_nk*cutoff_horizon_n[:, None]
            self.eval_spline(ts_nk, calculate_speeds=True)

        return tf.cast(valid_idxs, tf.int32)
    
    def check_dynamic_feasibility(self, speed_max_system, angular_speed_max_system, horizon_s):
        """Checks whether the current computed spline can be executed in time <= horizon_s (specified in seconds)
        while respecting max speed and angular speed constraints. Returns the batch indices of all valid splines."""
        
        # Compute the minimum horizon required to execute the spline while ensuring dynamic feasibility
        required_horizon_n1 = self.compute_dynamically_feasible_horizon(speed_max_system, angular_speed_max_system)
        
        # Compute the valid splines
        valid_idxs_n = tf.where(required_horizon_n1 <= horizon_s)[:, 0]
        return tf.cast(valid_idxs_n, tf.int32)
    
    def compute_dynamically_feasible_horizon(self, speed_max_system, angular_speed_max_system):
        """
        Compute the horizon (in seconds) such that the computed spline respect the speed and angular
        speed at all times.
        Speed assumed to be in [0, speed_max_system]
        Angular speed assumed to be in [-angular_speed_max_system, angular_speed_max_system]
        """
        # Compute the horizon required to make sure that we satisfy the speed constraints at all times
        max_speed_n1 = tf.reduce_max(self.speed_nk1(), axis=1)
        required_horizon_speed_n1 = self.final_times_n1 * max_speed_n1/speed_max_system

        # Compute the horizon required to make sure that we satisfy the angular speed constraints at all times
        max_angular_speed_n1 = tf.reduce_max(tf.abs(self.angular_speed_nk1()), axis=1)
        required_horizon_angular_speed_n1 = self.final_times_n1 * max_angular_speed_n1 / angular_speed_max_system
        
        # Compute the horizon required to make sure that we satisfy all control constraints at all times
        return tf.maximum(required_horizon_speed_n1, required_horizon_angular_speed_n1)
    
    def rescale_spline_horizon_to_dynamically_feasible_horizon(self, speed_max_system, angular_speed_max_system):
        """
        Rescale the spline horizon to a new horizon without recomputing the spline coefficients.
        """
        # Compute the minimum horizon required to execute the spline while ensuring dynamic feasibility
        required_horizon_n1 = self.compute_dynamically_feasible_horizon(speed_max_system, angular_speed_max_system)
        
        # Rescale the speed and angular velocity to be consistent with the new horizon
        self.rescale_velocity_and_acceleration(self.final_times_n1, required_horizon_n1)
        
        # Reset the final times
        self.final_times_n1 = required_horizon_n1
        
    def find_trajectories_within_a_horizon(self, horizon_s):
        """
        Find the indices of splines whose final time is within the horizon [0, horizon_s].
        """
        valid_idxs_n = tf.where(self.final_times_n1 <= horizon_s)[:, 0]
        return tf.cast(valid_idxs_n, tf.int32)

    #TODO: Probably Delete this
    @staticmethod
    def check_start_goal_equivalence(start_config_old, goal_config_old,
                                     start_config_new, goal_config_new):
        """ A utility function that checks whether start_config_old,
        goal_config_old imply the same spline constraints as those implied by
        start_config_new, goal_config_new. Useful for checking that a
        precomputed spline on the old start and goal will work on new start
        and goal."""
        start_old_pos_nk2 = start_config_old.position_nk2()
        start_old_heading_nk1 = start_config_old.heading_nk1()
        start_old_speed_nk1 = start_config_old.speed_nk1()

        start_new_pos_nk2 = start_config_new.position_nk2()
        start_new_heading_nk1 = start_config_new.heading_nk1()
        start_new_speed_nk1 = start_config_new.speed_nk1()

        start_pos_match = (tf.norm(start_old_pos_nk2-start_new_pos_nk2).numpy() == 0.0)
        start_heading_match = (tf.norm(start_old_heading_nk1-start_new_heading_nk1).numpy() == 0.0)
        start_speed_match = (tf.norm(start_old_speed_nk1-start_new_speed_nk1).numpy() == 0.0)

        start_match = (start_pos_match and start_heading_match and
                       start_speed_match)

        # Check whether they are the same object
        if goal_config_old is goal_config_new:
            return start_match
        else:
            goal_old_pos_nk2 = goal_config_old.position_nk2()
            goal_old_heading_nk1 = goal_config_old.heading_nk1()
            goal_old_speed_nk1 = goal_config_old.speed_nk1()

            goal_new_pos_nk2 = goal_config_new.position_nk2()
            goal_new_heading_nk1 = goal_config_new.heading_nk1()
            goal_new_speed_nk1 = goal_config_new.speed_nk1()

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

    def render(self, axs, batch_idx=0, freq=4, plot_heading=False,
               plot_velocity=False, label_start_and_end=True):
        super().render(axs, batch_idx, freq, plot_heading=plot_heading,
                       plot_velocity=plot_velocity,
                       label_start_and_end=label_start_and_end, name='Spline')
