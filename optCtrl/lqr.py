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
    Discrete time finite horizon LQR solver
    """
    def __init__(self, T, dynamics, cost):

        """
        T:              Length of horizon
        dynamics:       Discrete time plant dynamics, can be nonlinear
        cost:           instantaneous cost function; the terminal cost can be defined by judging the time index
        """

        self.T = T
        self.plant_dyn = dynamics
        self.cost = cost

        """
        Gradient of dynamics and costs with respect to state/control
        """
        self.plant_dyn_dx = None  # Df/Dx
        self.plant_dyn_du = None  # Df/Du
        self.cost_dx = None  # Dl/Dx
        self.cost_du = None  # Dl/Du
        self.cost_dxx = None  # D2l/Dx2
        self.cost_duu = None  # D2l/Du2
        self.cost_dux = None  # D2l/DuDx

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
        _, J = self.cost.compute_trajectory_cost(trajectory)
        return J

    def lqr(self, x0, trajectory, verbose=True):
        """
        Perform the iLQR iterations.
        x0:             Initial state
        trajectory:     The trajectory around which to linearize
        verbose:        Whether to print the status or not
        """
        with tf.name_scope('lqr'):
            # initialize the regularization term
            self.reg = 1

            # initialize current trajectory cost
            J_opt = self.evaluate_trajectory_cost(trajectory)
            J_hist = [J_opt]

            k_array, K_array = self.back_propagation(trajectory)
            trajectory_new = self.apply_control(x0, trajectory, k_array, K_array)
            # evaluate the cost of this trial
            J_new = self.evaluate_trajectory_cost(trajectory_new)
            J_opt = J_new

            J_hist.append(J_opt)

            # prepare result dictionary
            res_dict = {
                'J_hist': J_hist,
                'trajectory_opt': trajectory_new,
                'k_array_opt': k_array,
                'K_array_opt': K_array
            }

            return res_dict

    def apply_control(self, x0_n1d, trajectory, k_array, K_array):
        """
        apply the derived control to the error system to derive new x and u arrays
        """
        with tf.name_scope('apply_control'):
            assert(len(x0_n1d.shape) == 3) #[n,1,x_dim]
            angle_dims = self.plant_dyn._angle_dims
            n = x0_n1d.shape[0].value
            u_nkf = tf.zeros((n, 0, self.plant_dyn._u_dim), dtype=tf.float32)
            x_ref_nkd, u_ref_nkf = self.plant_dyn.parse_trajectory(trajectory)
            x_nkd = x0_n1d*1.
            x_tp1_n1d = x0_n1d*1.
            for t in range(self.T):
                x_ref_n1d, u_ref_n1f = x_ref_nkd[:,t:t+1], u_ref_nkf[:,t:t+1]
                error_t = x_tp1_n1d - x_ref_n1d
                error_t = tf.concat([error_t[:,:, :angle_dims],
                                        angle_normalize(error_t[:,:, angle_dims:angle_dims+1])], axis=2)
                fdback = tf.matmul(K_array[t], tf.transpose(error_t, perm=[0,2,1]))
                u_n1f = u_ref_n1f + tf.transpose(k_array[t] + fdback, perm=[0,2,1])
                x_tp1_n1d = self.fwdSim(x_tp1_n1d, u_n1f)
                u_nkf = tf.concat([u_nkf, u_n1f], axis=1)
                x_nkd = tf.concat([x_nkd, x_tp1_n1d], axis=1)

            trajectory = self.plant_dyn.assemble_trajectory(x_nkd, u_nkf, zero_pad_u=True)
            return trajectory

    def forward_propagation(self, x0, u_array):
        """
        Apply the forward dynamics to have a trajectory starting from x0 by applying u
        u_array is an array of control signal to apply
        """
        trajX_array, trajU_array, tau = self.plant_dyn.apply_controlSeq(x0, controlSeq=u_array.T,
                                                                        dynamics_sim=self.fwdSim)
        return trajX_array[:, :, 0].T

    def back_propagation(self, trajectory):
        """
        Back propagation along the given state and control trajectories to solve the Riccati equations for the error 
        system (delta_x, delta_u, t).
        Need to approximate the dynamics/costs along the given trajectory. 
        Dynamics needs a time-varying first-order approximation.
        Costs need time-varying second-order approximation.
        """
        with tf.name_scope('back_prop'):
            angle_dims = self.plant_dyn._angle_dims
            lqr_sys = self.build_lqr_system(trajectory)
            x_nkd, u_nkf = self.plant_dyn.parse_trajectory(trajectory)

            # k and K
            fdfwd = [None] * self.T
            fdbck_gain = [None] * self.T

            # initialize with the terminal cost parameters to prepare the backpropagation
            Vxx = lqr_sys['dldxx'][:,-1]
            Vx = lqr_sys['dldx'][:,-1]
            Vx = Vx[:,:,None]
            for t in reversed(range(self.T)):
                error_t = lqr_sys['f'][:,t]-x_nkd[:,t+1]
                error_t = tf.concat([error_t[:, :angle_dims],
                                    angle_normalize(error_t[:, angle_dims:angle_dims+1])], axis=1)
                error_t = error_t[:,:,None] 
                dfdx, dfdu = lqr_sys['dfdx'][:,t], lqr_sys['dfdu'][:,t]
                dfdx_T, dfdu_T = tf.transpose(dfdx, perm=[0,2,1]), tf.transpose(dfdu, perm=[0,2,1]) #transpose for matrix mult
                dfdx_T_dot_Vxx ,dfdu_T_dot_Vxx =  tf.matmul(dfdx_T, Vxx), tf.matmul(dfdu_T, Vxx)
               
                Qx = lqr_sys['dldx'][:,t][:,:,None] + tf.matmul(dfdx_T, Vx)+ tf.matmul(dfdx_T_dot_Vxx, error_t) 
                Qu = lqr_sys['dldu'][:,t][:,:,None] + tf.matmul(dfdu_T, Vx) + tf.matmul(dfdu_T_dot_Vxx, error_t)
                Qxx = lqr_sys['dldxx'][:,t] + tf.matmul(dfdx_T_dot_Vxx, dfdx) 
                Qux = lqr_sys['dldux'][:,t] + tf.matmul(dfdu_T_dot_Vxx, dfdx) 
                Quu = lqr_sys['dlduu'][:,t] + tf.matmul(dfdu_T_dot_Vxx, dfdu)

                # use regularized inverse for numerical stability
                inv_Quu = self.regularized_pseudo_inverse_(Quu, reg=self.reg)

                # get k and K
                fdfwd[t] = tf.matmul(-inv_Quu, Qu)
                fdbck_gain[t] = tf.matmul(-inv_Quu, Qux)
                fdbck_gain_T = tf.transpose(fdbck_gain[t], perm=[0,2,1])
                
                # update value function for the previous time step
                Vxx = Qxx - tf.matmul(tf.matmul(fdbck_gain_T, Quu), fdbck_gain[t])
                Vx = Qx - tf.matmul(tf.matmul(fdbck_gain_T, Quu), fdfwd[t])
            return fdfwd, fdbck_gain

    def build_lqr_system(self, trajectory):
        f_array = []
        dfdx_array = []
        dfdu_array = []
        dldx_array = []
        dldu_array = []
        dldxx_array = []
        dldux_array = []
        dlduu_array = []

        # First order linearization of the dynamics
        dfdx, dfdu, f = self.plant_dyn.affine_factors(trajectory)

        dldxx, dldxu, dlduu, dldx, dldu = self.cost.quad_coeffs(trajectory)
        dldux = tf.transpose(dldxu, perm=[0,1,3,2])

        lqr_sys = {
            'f': f,
            'dfdx': dfdx,
            'dfdu': dfdu,
            'dldx': dldx,
            'dldu': dldu,
            'dldxx': dldxx,
            'dlduu': dlduu,
            'dldux': dldux
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
            s,u,v = tf.svd(mat)
            s = tf.nn.relu(s) #truncate negative values
            #diag_s_inv = np.zeros((v.shape[0], u.shape[1]))
            #diag_s_inv[0:len(s), 0:len(s)] = np.diag(1. / (s + reg))
            #return v.dot(diag_s_inv).dot(u.T)
            raise NotImplementedError
            
