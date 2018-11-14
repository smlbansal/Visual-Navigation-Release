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


def main(cutoff=2.5):
    """Visualize the result of applying
    saved LQR feedback matrices from a start robot
    configuration. Converts the lqr feedback & reference trajectory
    to an egocentric frame for easy deployment on a real robot."""
    p = load_params()
    utils.mkdir_if_missing(egocentric_data_dir)
    utils.mkdir_if_missing(data_dir)

    filenames = os.listdir(data_dir) 
    filenames = list(filter(lambda x: '.pkl' in x, filenames))
    filenames.sort(key=lambda filename: int(filename.split('.')[0].split('_')[-1]))
    filenames = filenames
    N = len(filenames)
    sqrt_num_plots = int(np.ceil(np.sqrt(N)))
    fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                 (8, 8), (.4, .4))

    fig1, _, axs1 = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                   (8, 8), (.4, .4))

    axs = axs[::-1]
    axs1 = axs1[::-1]
    freq = 25
    for i, filename in enumerate(filenames):
        print(i)
        goal_num = int(filename.split('.')[0].split('_')[-1])
        full_filename = os.path.join(data_dir, filename)
        with open(full_filename, 'rb') as f:
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
        lqr_trajectory.render([axs[i]], name='Goal #{:d}'.format(goal_num), freq=freq)
        axs[i].set_xlim([0.0, 4.0])
        axs[i].set_ylim([0.0, 2.5])

        ref_trajectory_egocentric = dubins.to_egocentric_coordinates(start_config, ref_trajectory,
                                                                     mode='new')
        start_config_egocentric = dubins.to_egocentric_coordinates(start_config, start_config,
                                                                   mode='new')
        K_1kfd_egocentric = dubins.convert_K_to_egocentric_coordinates(start_config, K_1kfd,
                                                                       mode='new')
        lqr_trajectory = lqrSolver.apply_control(start_config_egocentric,
                                                 ref_trajectory_egocentric, k_1kf1,
                                                 K_1kfd_egocentric)
        lqr_trajectory.render([axs1[i]], name='Goal #{:d}'.format(goal_num), freq=freq)
        x, y = start_config.position_nk2()[0, 0].numpy()
        axs1[i].set_xlim([-x, -x+4.0])
        axs1[i].set_ylim([-y, -y+2.5])

        data_egocentric = {'k_1kf1': data['k_1kf1'],
                           'trajectory': ref_trajectory_egocentric.to_numpy_repr(),
                           'K_1kfd': K_1kfd_egocentric.numpy()}
        full_filename = os.path.join(egocentric_data_dir, filename)
        with open(full_filename, 'wb') as f:
            pickle.dump(data_egocentric, f)

    filename = os.path.join(data_dir, 'saved_lqr.png')
    fig.savefig(filename, bbox_inches='tight')

    filename = os.path.join(egocentric_data_dir, 'saved_lqr.png')
    fig1.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    tf.enable_eager_execution()
    plt.style.use('ggplot')
    main()
