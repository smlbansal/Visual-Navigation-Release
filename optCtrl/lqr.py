import tensorflow as tf
from utils.angle_utils import angle_normalize
# Algorithms from Somil Init trajectory optimizer, variant of the classic LQR
#######################################################################################################################
# Steps for solving Trajectory Optimizer

# 0.   Initialization x0, xf, u, l(cost function), f(dynamics), Q(quadraticized cost), V(value function)
# 1.   Derivatives
#      Given a nominal sequence (x, u, i) computes first and second derivatives of l and f, respectively.
# 2-1. BackWard Pass
#      Iterating equations related to Q, K and V for decreasing i = N-1, ... 1.
# 2-2. Condition Hold
#      If non-PD(Positive Definite) Q_uu is encountered, fix using regularization, and restart the backward pass or,
#      decrease regularization if successful.
#######################################################################################################################


class LQRSolver:
    """
    Discrete time finite horizon LQR solver. Dimensions used throughout this
    class are:
            n- batch size
            d- x_dim
            f- u_dim
            T- self.T (LQR Planning Horizon)

    """
    def __init__(self, T, dynamics, cost):

        """
        T:              Length of horizon
        dynamics:       Discrete time plant dynamics, can be nonlinear
        cost:           instantaneous cost function
        """

        self.T = T
        self.plant_dyn = dynamics
        self.cost = cost

        # Forward simulation function for dynamics
        if dynamics.isStochastic:
            self.fwdSim = dynamics.simulate_mean
        else:
            self.fwdSim = dynamics.simulate

        # adapting the regularizer for the input penalty
        self.reg = .1
        self.reg_max = 1000
        self.reg_min = 1e-6
        self.reg_factor = 10

        # Whether to take the full inverse or use SVD during the Bellman backup
        self.inv = True
        return

    def evaluate_trajectory_cost(self, trajectory):
        """
        Compute the cost incurred by a trajectory.
        """
        # Note(Somil): Add the dimensions of J?
        _, J = self.cost.compute_trajectory_cost(trajectory)
        return J

    def lqr(self, start_config, trajectory, verbose=True, sim_mode='ideal'):
        """
        Perform the iLQR iterations.
        start_config:             Initial system configuration
        trajectory:     The trajectory around which to linearize
        verbose:        Whether to print the status or not
        """
        with tf.name_scope('lqr'):
            # initialize the regularization term
            self.reg = 1

            # initialize current trajectory cost
            J_opt_n = self.evaluate_trajectory_cost(trajectory)
            J_hist = [J_opt_n]

            # k_array, and K_array are tensors of dimension
            # (n, self.T-1, u_dim, 1) and (n, self.T-1, u_dim, x_dim) respectively
            k_array_nTf1, K_array_nTfd = self.back_propagation(trajectory)
            trajectory_new = self.apply_control(start_config, trajectory,
                                                k_array_nTf1, K_array_nTfd,
                                                sim_mode=sim_mode)

            # evaluate the cost of this trial
            J_new_n = self.evaluate_trajectory_cost(trajectory_new)
            J_hist.append(J_new_n)

            # prepare result dictionary
            res_dict = {
                'J_hist': J_hist,
                'trajectory_opt': trajectory_new,
                'k_opt_nkf1': k_array_nTf1,
                'K_opt_nkfd': K_array_nTfd
            }
            return res_dict

    def apply_control(self, start_config, trajectory,
                      k_array_nTf1, K_array_nTfd,
                      sim_mode='ideal'):
        """
        apply the derived control to the error system to derive a new
        trajectory. Here k_array_nTf1 and K_aaray_nTfd are
        tensors of dimension (n, self.T-1, f, 1) and (n, self.T-1, f, d) respectively.
        """
        with tf.name_scope('apply_control'):
            x0_n1d, _ = self.plant_dyn.parse_trajectory(start_config)
            assert(len(x0_n1d.shape) == 3)  # [n,1,x_dim]
            angle_dims = self.plant_dyn._angle_dims
            actions = []
            states = [x0_n1d*1.]
            x_ref_nkd, u_ref_nkf = self.plant_dyn.parse_trajectory(trajectory)
            x_next_n1d = x0_n1d*1.
            for t in range(self.T):
                x_ref_n1d, u_ref_n1f = x_ref_nkd[:, t:t+1], u_ref_nkf[:, t:t+1]
                error_t_n1d = x_next_n1d - x_ref_n1d
            
                # TODO: Currently calling numpy() here as tfe.DEVICE_PLACEMENT_SILENT
                # is not working to place non-gpu ops (i.e. mod) on the cpu
                # turning tensors into numpy arrays is a hack around this.
                error_t_n1d = tf.concat([error_t_n1d[:, :, :angle_dims],
                                         angle_normalize(error_t_n1d[:, :,
                                                                     angle_dims:angle_dims+1].numpy()),
                                         error_t_n1d[:, :, angle_dims+1:]],
                                        axis=2)
                fdback_nf1 = tf.matmul(K_array_nTfd[:, t],
                                       tf.transpose(error_t_n1d, perm=[0, 2, 1]))
                u_n1f = u_ref_n1f + tf.transpose(k_array_nTf1[:, t] + fdback_nf1,
                                                 perm=[0, 2, 1])
                x_next_n1d = self.fwdSim(x_next_n1d, u_n1f, mode=sim_mode)
                actions.append(u_n1f)
                states.append(x_next_n1d)
            u_nkf = tf.concat(actions, axis=1)
            x_nkd = tf.concat(states, axis=1)
            trajectory = self.plant_dyn.assemble_trajectory(x_nkd,
                                                            u_nkf,
                                                            pad_mode='repeat')
            return trajectory

    def back_propagation(self, trajectory):
        """
        Back propagation along the given trajectory (state and control),
        solving the Riccati equations for the error
        system (delta_x, delta_u, t).
        Need to approximate the dynamics/costs along the given trajectory.
        Dynamics needs a time-varying first-order approximation.
        Costs need time-varying second-order approximation. """
        with tf.name_scope('back_prop'):
            angle_dims = self.plant_dyn._angle_dims
            lqr_sys = self.build_lqr_system(trajectory)
            x_nkd, u_nkf = self.plant_dyn.parse_trajectory(trajectory)

            # k (feedforward) and K (feedback) are lists of length self.T
            # where each element is a tensor of dimension (n, f, 1)
            # and (n, f, d) respectively.
            fdfwd_Tnf1 = [None] * self.T
            fdbck_gain_Tnfd = [None] * self.T

            # initialize with the terminal cost parameters
            # to prepare the backpropagation
            Vxx_ndd = lqr_sys['dldxx_nkdd'][:, -1]
            Vx_nd1 = lqr_sys['dldx_nkd'][:, -1, :, None]

            # TODO: Currently calling numpy() here as tfe.DEVICE_PLACEMENT_SILENT
            # is not working to place non-gpu ops (i.e. mod) on the cpu
            # turning tensors into numpy arrays is a hack around this.

            for t in reversed(range(self.T)):
                error_t_nd = lqr_sys['f_nkd'][:, t]-x_nkd[:, t+1]
                error_t_nd = tf.concat([error_t_nd[:, :angle_dims],
                                        angle_normalize(error_t_nd[:,
                                                                   angle_dims:angle_dims+1].numpy()),
                                        error_t_nd[:, angle_dims+1:]],
                                       axis=1)
                error_t_nd1 = error_t_nd[:, :, None]

                dfdx_ndd = lqr_sys['dfdx_nkdd'][:, t]
                dfdu_ndf = lqr_sys['dfdu_nkdf'][:, t]
                dfdx_T_ndd = tf.transpose(dfdx_ndd, perm=[0, 2, 1])
                dfdu_T_ndf = tf.transpose(dfdu_ndf, perm=[0, 2, 1])

                dfdx_T_dot_Vxx_ndd = tf.matmul(dfdx_T_ndd, Vxx_ndd)
                dfdu_T_dot_Vxx_nfd = tf.matmul(dfdu_T_ndf, Vxx_ndd)

                Qx_nd1 = (lqr_sys['dldx_nkd'][:, t][:, :, None] + tf.matmul(dfdx_T_ndd, Vx_nd1)
                          + tf.matmul(dfdx_T_dot_Vxx_ndd, error_t_nd1))
                Qu_nf1 = (lqr_sys['dldu_nkf'][:, t][:, :, None] + tf.matmul(dfdu_T_ndf, Vx_nd1)
                          + tf.matmul(dfdu_T_dot_Vxx_nfd, error_t_nd1))
                Qxx_ndd = lqr_sys['dldxx_nkdd'][:, t] + tf.matmul(dfdx_T_dot_Vxx_ndd, dfdx_ndd)
                Qux_nfd = lqr_sys['dldux_nkfd'][:, t] + tf.matmul(dfdu_T_dot_Vxx_nfd, dfdx_ndd)
                Quu_nff = lqr_sys['dlduu_nkff'][:, t] + tf.matmul(dfdu_T_dot_Vxx_nfd, dfdu_ndf)

                # use regularized inverse for numerical stability
                inv_Quu_nff = self.regularized_pseudo_inverse_(Quu_nff, reg=self.reg)

                # get k and K
                fdfwd_Tnf1[t] = tf.matmul(-inv_Quu_nff, Qu_nf1)
                fdbck_gain_Tnfd[t] = tf.matmul(-inv_Quu_nff, Qux_nfd)
                fdbck_gain_nfd = tf.transpose(fdbck_gain_Tnfd[t], perm=[0, 2, 1])

                # update value function for the previous time step
                Vxx_ndd = Qxx_ndd - tf.matmul(tf.matmul(fdbck_gain_nfd, Quu_nff),
                                              fdbck_gain_Tnfd[t])
                Vx_nd1 = Qx_nd1 - tf.matmul(tf.matmul(fdbck_gain_nfd, Quu_nff), fdfwd_Tnf1[t])

            # Stack the outer time dimension as dimension 1
            # in the tensors
            fdfwd_nTf1 = tf.stack(fdfwd_Tnf1, axis=1)
            fdbck_gain_nTfd = tf.stack(fdbck_gain_Tnfd, axis=1)
            return fdfwd_nTf1, fdbck_gain_nTfd

    def build_lqr_system(self, trajectory):
        """Given a trajectory returns the first order
        approximation of dynamics(f) and second order
        approximation of cost/loss(l) needed for LQR"""

        # First order approximation of the dynamics (f)
        dfdx_nkdd, dfdu_nkdf, f_nkd = self.plant_dyn.affine_factors(trajectory)

        # Second order approximation of cost/loss (l)
        dldxx_nkdd, dldxu_nkdf, dlduu_nkff, dldx_nkd, dldu_nkf = self.cost.quad_coeffs(trajectory)
        dldux_nkfd = tf.transpose(dldxu_nkdf, perm=[0, 1, 3, 2])

        lqr_sys = {
            'f_nkd': f_nkd,
            'dfdx_nkdd': dfdx_nkdd,
            'dfdu_nkdf': dfdu_nkdf,
            'dldx_nkd': dldx_nkd,
            'dldu_nkf': dldu_nkf,
            'dldxx_nkdd': dldxx_nkdd,
            'dlduu_nkff': dlduu_nkff,
            'dldux_nkfd': dldux_nkfd
        }
        return lqr_sys

    def regularized_pseudo_inverse_(self, mat, reg=1e-5):
        """
        Use SVD to realize pseudo inverse by perturbing the singularity values
        to ensure its positive-definite properties
        """
        if self.inv:
            return tf.matrix_inverse(mat)
        else:
            raise NotImplementedError
