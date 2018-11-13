import pickle
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
from trajectory.trajectory import Trajectory, SystemConfig
from optCtrl.lqr import LQRSolver

data_dir = './logs/simulator/lqr_data/'


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
    p = load_params() 
    files = os.listdir(data_dir)
    files.sort()
    N = len(files)
    sqrt_num_plots = int(np.ceil(np.sqrt(N)))
    fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                 (8, 8), (.4, .4))
    axs = axs[::-1]
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
        import pdb; pdb.set_trace()
        start_config = SystemConfig.init_config_from_trajectory_time_index(ref_trajectory, 0)
        lqrSolver = LQRSolver(T, system=dubins, cost=None) 
    test = 5


if __name__ == '__main__':
    main()
