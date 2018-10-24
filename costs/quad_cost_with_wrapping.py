import tensorflow as tf
from costs.cost import DiscreteCost
from utils.angle_utils import angle_normalize


class QuadraticRegulatorRef(DiscreteCost):
    """
    Creates a quadratic cost of the form 0.5*[x-x_ref(t) u-u_ref(t)]*C*[x-x_ref(t) u-u_ref(t)]^T +
    c*[x-x_ref(t) u-u_ref(t)]^T for every time step. However, some dimensions
    are angles, which are wrapped in the cost.
    """

    def __init__(self, trajectory_ref, C_gg, c_g, system):
        """
        :param: trajectory_ref: A reference trajectory against which to penalize
                C_gg, c_g: Quadratic and linear penalties (of dimension g where
                            g = d + f)
                angle_dims: index array which specifies the dimensions of the
                state that corresponds to angles and should be wrapped.
        """

        # Used for parsing trajectory objects into tensors of states and actions
        self.system = system

        self._x_dim, self._u_dim = system._x_dim, system._u_dim  # d, f
        self.angle_dims = system._angle_dims
        self.update_trajectory_ref_and_cost(trajectory_ref, C_gg, c_g)
        super().__init__(x_dim=self._x_dim, u_dim=self._u_dim)

        self.isTimevarying = True
        self.isNonquadratic = False

    def update_trajectory_ref_and_cost(self, trajectory_ref, C_gg, c_g):
        """Update the reference trajectory and quadratic/linear penalites of the
        cost function. Allows a cost function object to be reused over multiple trajectories
        of different shapes."""
        x_dim, u_dim = self._x_dim, self._u_dim
        # Check dimensions
        assert ((tf.reduce_all(tf.equal(C_gg[:x_dim, x_dim:],
                                        tf.transpose(C_gg[x_dim:, :x_dim]))).numpy()))
        assert ((x_dim + u_dim) == C_gg.shape[0].value
                == C_gg.shape[1].value == c_g.shape[0].value)

        self.trajectory_ref = trajectory_ref
        n, k, g = trajectory_ref.n, trajectory_ref.k, C_gg.shape[0]
        self._C_nkgg = tf.broadcast_to(C_gg, (n, k, g, g))
        self._c_nkg = tf.broadcast_to(c_g, (n, k, g))

    def compute_trajectory_cost(self, trajectory, trials=1):
        with tf.name_scope('compute_traj_cost'):
            z_nkg = self.construct_z(trajectory)
            C_nkgg, c_nkg = self._C_nkgg, self._c_nkg
            Cz_nkg = tf.squeeze(tf.matmul(C_nkgg, z_nkg[:, :, :, None]))
            zCz_nk = tf.reduce_sum(z_nkg*Cz_nkg, axis=2)
            cz_nk = tf.reduce_sum(c_nkg*z_nkg, axis=2)
            cost_nk = .5*zCz_nk + cz_nk
            return cost_nk, tf.reduce_sum(cost_nk, axis=1)

    def quad_coeffs(self, trajectory, t=None):
        # Return terms H_xx, H_xu, H_uu, J_x, J_u
        with tf.name_scope('quad_coeffs'):
            H_nkgg = self._C_nkgg
            J_nkg = self._c_nkg
            z_nkg = self.construct_z(trajectory)
            Hz_nkg = tf.squeeze(tf.matmul(H_nkgg, z_nkg[:, :, :, None]),
                                axis=-1)
            return H_nkgg[:, :, :self._x_dim, :self._x_dim], \
                   H_nkgg[:, :, :self._x_dim, self._x_dim:], \
                   H_nkgg[:, :, self._x_dim:, self._x_dim:], \
                   J_nkg[:, :, :self._x_dim] + Hz_nkg[:, :, :self._x_dim], \
                   J_nkg[:, :, self._x_dim:] + Hz_nkg[:, :, self._x_dim:]

    def construct_z(self, trajectory):
        """ Input: A trajectory with x_dim =d and u_dim=f
            Output: z_nkg - a tensor of size n,k,g where g=d+f
        """
        with tf.name_scope('construct_z'):
            x_nkd, u_nkf = self.system.parse_trajectory(trajectory)
            x_ref_nkd, u_ref_nkf = self.system.parse_trajectory(self.trajectory_ref)
            delx_nkd = x_nkd - x_ref_nkd
            delu_nkf = u_nkf - u_ref_nkf
            z_nkg = tf.concat([delx_nkd[:, :, :self.angle_dims],
                               angle_normalize(delx_nkd[:, :, self.angle_dims:self.angle_dims+1]),
                               delx_nkd[:, :, self.angle_dims+1:],
                               delu_nkf], axis=2)
            return z_nkg
