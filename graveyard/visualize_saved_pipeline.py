import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import utils
import pickle
import os

from trajectory.trajectory import Trajectory, SystemConfig


dirname = './data/control_pipelines_old/control_pipeline_v0/planning_horizon_120_dt_0.05/uniform_grid_n_21734_theta_bins_21_bound_min_0.00_-2.50_-1.57_bound_max_2.50_2.50_1.57'
filename = '_velocity_0.040.pkl'
dv = .01

def visualize_pipeline(N):
    filenames = list(filter(lambda x: 'velocity' in x, os.listdir(dirname)))
    filenames.sort()

    for filename in filenames:
        v0 = float('.{:s}'.format(filename.split('.')[-2])) 
        filename2 = os.path.join(dirname, filename)
        with open(filename2, 'rb') as f:
            data = pickle.load(f)
        lqr_trajectories = Trajectory.init_from_numpy_repr(**data['lqr_trajectories'])
        lqr_speed_n1 = lqr_trajectories.speed_nk1()[:, 0, :]
        dist = np.abs(v0-lqr_speed_n1)[:, 0]
        percent_correct = 100.*np.sum(dist <= dv/2.)/ len(dist)
        percent_incorrect = 100.*np.sum(dist > dv/2.)/ len(dist)
        max_v_error = np.max(dist)
        print('V: {:.3f}, {:.3f}% Correct Bin, {:.3f}% Incorrect Bin, {:.3f} max V error'.format(v0, percent_correct,
                                                                             percent_incorrect,
                                                                             max_v_error))
    velocity = input('Input desired velocity here: ') 
    velocity = float(velocity)
    filename = '_velocity_{:1.3f}.pkl'.format(velocity)
    
    fig, _, axs = utils.subplot2(plt, (2*N, 4), (8, 8), (.4, .4))
    axs = axs[::-1]
    filename2 = os.path.join(dirname, filename)
    with open(filename2, 'rb') as f:
        data = pickle.load(f)
   
    lqr_trajectories = Trajectory.init_from_numpy_repr(**data['lqr_trajectories'])
    spline_trajectories = Trajectory.init_from_numpy_repr(**data['spline_trajectories'])
    waypt_configs = SystemConfig.init_from_numpy_repr(**data['waypt_configs'])

    v0 = float('.{:s}'.format(filename.split('.')[-2]))
    print(v0)
    spline_speed_n1 = data['start_speeds']
    lqr_speed_n1 = lqr_trajectories.speed_nk1()[:, 0, :]
    dist = np.abs(v0-lqr_speed_n1)[:, 0]
    idxs = np.argpartition(dist, -N, axis=0)[-N:]

    for i, idx in enumerate(idxs):
        axs0, axs1 = axs[2*i*4: (2*i+1)*4], axs[2*i*4+4: (2*i+1)*4+4]
        spline_trajectories.render(axs0, batch_idx=idx, plot_heading=True, plot_velocity=True,
                                label_start_and_end=True, name='Spline')
        lqr_trajectories.render(axs1, batch_idx=idx, plot_heading=True, plot_velocity=True,
                                label_start_and_end=True, name='LQR')
    filename = './tmp/problematic_trajectories_v0_{:.3f}.png'.format(velocity)
    fig.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    tf.enable_eager_execution()
    visualize_pipeline(2)
    
