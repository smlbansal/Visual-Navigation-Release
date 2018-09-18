import tensorflow as tf
import numpy as np
from costs.cost import DiscreteCost

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class QuadraticRegulatorRef(DiscreteCost):
    """ 
    Creates a quadratic cost of the form 0.5*[x-x_ref(t) u-u_ref(t)]*C*[x-x_ref(t) u-u_ref(t)]^T + 
    c*[x-x_ref(t) u-u_ref(t)]^T for every time step. However, some dimensions are angles, which are wrapped in 
    the cost.
    """

    def __init__(self, trajectory_ref, C, c, system):
        """
        :param: x_ref, u_ref: state and controller reference trajectories
                C, c: Quadratic and linear penalties
                angle_dims: index array which specifies the dimensions of the state that corresponds to angles and 
                should be wrapped.
        """
      
        self.system = system 
        x_dim, u_dim = system._x_dim, system._u_dim #d,f
        assert (tf.reduce_all(tf.equal(C[:x_dim, x_dim:], tf.transpose(C[x_dim:, :x_dim]))).numpy())
        assert (x_dim + u_dim) == C.shape[0].value == C.shape[1].value == c.shape[0].value
        x_ref_nkd, u_ref_nkf = system.parse_trajectory(trajectory_ref)
        self._x_ref_nkd = x_ref_nkd
        self._u_ref_nkf = u_ref_nkf 
        self._x_dim = x_dim
        self._u_dim = u_dim
        self._z_ref_nkg = tf.concat([x_ref_nkd, u_ref_nkf],axis=2) 
        self.angle_dims = system._angle_dims
        self._C_nkgg = C[None,None] + 0.*self._z_ref_nkg[:,:,:,None]
        self._c_nkg = c[None,None] + 0. *self._z_ref_nkg
        super().__init__(x_dim=self._x_dim, u_dim=self._u_dim)

        self.isTimevarying = False
        self.isNonquadratic = False

    def compute_trajectory_cost(self, trajectory):
        with tf.name_scope('compute_traj_cost'):
            z_nkg = self.construct_z(trajectory)
            C_nkgg, c_nkg = self._C_nkgg, self._c_nkg
            Cz_nkg = self.matrix_vector_prod_nkgg(C_nkgg, z_nkg) 
            zCz_nk = tf.reduce_sum(z_nkg*Cz_nkg, axis=2)
            cz_nk = tf.reduce_sum(c_nkg*z_nkg, axis=2)
            cost = .5*zCz_nk + cz_nk
            return cost, tf.reduce_sum(cost, axis=1)

    def quad_coeffs(self, trajectory, t=None):
        # Return terms H_xx, H_xu, H_uu, J_x, J_u
        with tf.name_scope('quad_coeffs'):
            H_nkgg = self._C_nkgg
            J_nkg = self._c_nkg
            z_nkg = self.construct_z(trajectory)
            Hz_nkg = self.matrix_vector_prod_nkgg(H_nkgg, z_nkg) 
            return H_nkgg[:,:,:self._x_dim, :self._x_dim], \
                   H_nkgg[:,:,:self._x_dim, self._x_dim:], \
                   H_nkgg[:,:,self._x_dim:, self._x_dim:], \
                   J_nkg[:,:,:self._x_dim] + Hz_nkg[:,:,:self._x_dim], \
                   J_nkg[:,:,self._x_dim:] + Hz_nkg[:,:,self._x_dim:]

    def construct_z(self, trajectory):
        """ Input: A trajectory with x_dim =d and u_dim=f
            Output: z_nkg - a tensor of size n,k,g where g=d+f
        """
        with tf.name_scope('construct_z'):
            x_nkd, u_nkf = self.system.parse_trajectory(trajectory)
            delx_nkd = x_nkd - self._x_ref_nkd 
            delu_nkf = u_nkf - self._u_ref_nkf
            z_nkg = tf.concat([delx_nkd[:,:,:self.angle_dims],
                            angle_normalize(delx_nkd[:,:,self.angle_dims:self.angle_dims+1]),
                            delx_nkd[:,:,self.angle_dims+1:],
                            delu_nkf], axis=2)
            return z_nkg

    def matrix_vector_prod_nkgg(self, C_nkgg, z_nkg):
        """Input: C_nkgg, z_nkg a matrix and vector
        Calculates the matrix vector dot product 
            C_gg*z_g
        broadcast across the n and k dimensions
        """
        with tf.name_scope('dot_product'):
            zr_nkgg = z_nkg[:,:,None] + 0.*C_nkgg
            Cz_dot_prod_nkg = tf.reduce_sum(C_nkgg * zr_nkgg, axis=3)
            return Cz_dot_prod_nkg

