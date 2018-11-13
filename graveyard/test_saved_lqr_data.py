import pickle
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
from trajectory.trajectory import Trajectory, SystemConfig
from optCtrl.lqr import LQRSolver

data_dir = './logs/simulator/lqr_data/'
egocentric_data_dir = './logs/simulator/lqr_data_egocentric/'

def save_lqr_data(filename, trajectory, controllers):
        """ Saves the LQR controllers (K, k) used to track the current vehicle
        trajectory as well as the current vehicle trajectory."""
        data = {'trajectory' : trajectory.to_numpy_repr(),
                'K_1kfd' : controllers['K_1kfd'].numpy(),
                'k_1kf1' : controllers['k_1kf1'].numpy()}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

def load_params():
    from dotmap import DotMap
    from systems.dubins_v2 import DubinsV2
    from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
    
    p = DotMap()
    # System Dynamics params
    p.system_dynamics_params = DotMap(system=DubinsV2,
                                      dt=.05,
                                      v_bounds=[0.0, .6],
                                      w_bounds=[-1.1, 1.1])

    # LQR setting parameters
    p.lqr_params = DotMap(cost_fn=QuadraticRegulatorRef,
                          quad_coeffs=np.array(
                              [1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                          linear_coeffs=np.zeros((5), dtype=np.float32))
    return p


def main():
    """Visualize the result of applying
    saved LQR feedback matrices from a start robot
    configuration. Good visual sanity check for checking
    LQR coordinate transforms."""
    p = load_params()
    utils.mkdir_if_missing(egocentric_data_dir)
    utils.mkdir_if_missing(data_dir)

    files = os.listdir(data_dir)[:5]
    files.sort()
    N = len(files)
    sqrt_num_plots = int(np.ceil(np.sqrt(N)))
    fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                 (8, 8), (.4, .4))

    fig1, _, axs1 = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                   (8, 8), (.4, .4))

    axs = axs[::-1]
    axs1 = axs1[::-1]
    for i, filename in enumerate(files):
        filename = os.path.join(data_dir, filename)
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        ref_trajectory = Trajectory.init_from_numpy_repr(**data['trajectory'])
        K_1kfd = tf.constant(data['K_1kfd'])
        k_1kf1 = tf.constant(data['k_1kf1'])
       
        ps = p.system_dynamics_params
        dubins = ps.system(dt=ps.dt, params=ps) 
        T = ref_trajectory.k - 1
        lqrSolver = LQRSolver(T, dynamics=dubins, cost=None)
        
        start_config = SystemConfig.init_config_from_trajectory_time_index(ref_trajectory, 0)
        lqr_trajectory = lqrSolver.apply_control(start_config, ref_trajectory, k_1kf1, K_1kfd)
        lqr_trajectory.render([axs[i]], name='Goal #{:d}'.format(i))
        axs[i].set_xlim([0.0, 8.0])
        axs[i].set_ylim([0.0, 8.0])

        import pdb; pdb.set_trace()
        ref_trajectory_egocentric = dubins.to_egocentric_coordinates(start_config, ref_trajectory,
                                                                    mode='new')
        start_config_egocentric = dubins.to_egocentric_coordinates(start_config, start_config,
                                                                   mode='new')
        K_1kfd_egocentric = dubins.to_egocentric_coordinates()
        lqr_trajectory = lqrSolver.apply_control(start_config_egocentric,
                                                 ref_trajectory_egocentric, k_1kf1,
                                                 K_1kfd_egocentric)
        lqr_trajectory.render([axs1[i]], name='Goal #{:d}'.format(i))
        axs1[i].set_xlim([0.0, 8.0])
        axs1[i].set_ylim([0.0, 8.0])

    filename = os.path.join(data_dir, 'saved_lqr.png')
    fig.savefig(filename, bbox_inches='tight')

    filename = os.path.join(egocentric_data_dir, 'saved_lqr.png')
    fig1.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    tf.enable_eager_execution()
    plt.style.use('ggplot')
    main()
