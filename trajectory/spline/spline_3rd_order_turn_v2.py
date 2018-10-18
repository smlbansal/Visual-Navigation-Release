from trajectory.spline.spline_3rd_order import Spline3rdOrder
import tensorflow as tf


class Spline3rdOrderTurnV2(Spline3rdOrder):
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
        
        with tf.name_scope('fit_spline'):
            t0_n1 = self.start_state.heading_nk1()[:, :, 0]
            tg_n1 = self.goal_state.heading_nk1()[:, :, 0]

            b1_n1 = t0_n1
            a1_n1 = tg_n1-t0_n1
            self.theta_coeffs_n12 = tf.stack([a1_n1, b1_n1], axis=2)
            
    def _eval_spline(self, ts_nk, calculate_speeds=True):
        """ Evaluates the spline on points in ts_nk
        Assumes ts is normalized to be in [0, 1.]
        """
        theta_coeffs_n12 = self.theta_coeffs_n12
        
        with tf.name_scope('eval_spline'):
            ts_n2k = tf.stack([ts_nk, tf.ones_like(ts_nk)], axis=1)
            theta_nk = tf.squeeze(tf.matmul(theta_coeffs_n12, ts_n2k), axis=1)

            start_pos_n12 = self.start_state.position_nk2()
            self._position_nk2 = tf.broadcast_to(start_pos_n12, (self.n, self.k, 2))
            self._heading_nk1 = theta_nk[:, :, None]
