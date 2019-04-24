import tensorflow as tf
from costs.cost import DiscreteCost
from utils.angle_utils import angle_normalize


class QuadraticRegulatorRef(DiscreteCost):
    """
    Creates a quadratic cost of the form 0.5*[x-x_ref(t) u-u_ref(t)]*C*[x-x_ref(t) u-u_ref(t)]^T +
    c*[x-x_ref(t) u-u_ref(t)]^T for every time step. However, some dimensions
    are angles, which are wrapped in the cost.
    """

    def __init__(self, trajectory_ref, system, params):
        """
        :param: trajectory_ref: A reference trajectory against which to penalize
                system: A system dynamics object (used for parsing trajectory objects)
                state that corresponds to angles and should be wrapped.
        """
        
        # Used for parsing trajectory objects into tensors of states and actions
        self.params = params
        self.system = system

        self._x_dim, self._u_dim = system._x_dim, system._u_dim  # d, f
        self.angle_dims = system._angle_dims

        self.trajectory_ref = trajectory_ref
        self.update_shape()
        super(QuadraticRegulatorRef, self).__init__(x_dim=self._x_dim, u_dim=self._u_dim)

        self.isTimevarying = True
        self.isNonquadratic = False

    def update_shape(self):
        """Update the shape of quadratic/linear penalites of the cost function based on the shape of
        self.trajectory_ref. Allows a cost function object to be reused with a trajectory object as it changes shape."""
        p = self.params

        x_dim, u_dim = self._x_dim, self._u_dim

        C_gg = tf.diag(p.quad_coeffs, name='lqr_coeffs_quad')
        c_g = tf.constant(p.linear_coeffs, name='lqr_coeffs_linear', dtype=tf.float32)
        # Check dimensions
        assert ((tf.reduce_all(tf.equal(C_gg[:x_dim, x_dim:], tf.transpose(C_gg[x_dim:, :x_dim]))).numpy()))
        assert ((x_dim + u_dim) == C_gg.shape[0].value == C_gg.shape[1].value == c_g.shape[0].value)

        trajectory_ref = self.trajectory_ref
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
            Hz_nkg = tf.squeeze(tf.matmul(H_nkgg, z_nkg[:, :, :, None]), axis=-1)
            return H_nkgg[:, :, :self._x_dim, :self._x_dim], \
                   H_nkgg[:, :, :self._x_dim, self._x_dim:], \
                   H_nkgg[:, :, self._x_dim:, self._x_dim:], \
                   J_nkg[:, :, :self._x_dim] + Hz_nkg[:, :, :self._x_dim], \
                   J_nkg[:, :, self._x_dim:] + Hz_nkg[:, :, self._x_dim:]

    # TODO: Currently calling numpy() here as tfe.DEVICE_PLACEMENT_SILENT is not working in the eager mode to place
    # non-gpu ops (i.e. mod) on the cpu turning tensors into numpy arrays is a hack around this.
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
                               angle_normalize(delx_nkd[:, :, self.angle_dims:self.angle_dims+1].numpy()),
                               delx_nkd[:, :, self.angle_dims+1:], delu_nkf], axis=2)
            return z_nkg
