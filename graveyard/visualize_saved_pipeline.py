import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import utils
import pickle
import os

from trajectory.trajectory import Trajectory, SystemConfig


dirname = './data/control_pipelines_dubins_v2/control_pipeline_v0/planning_horizon_120_dt_0.05'
filename = 'n_21734_theta_bins_21_bound_min_0.00_-2.50_-1.57_bound_max_2.50_2.50_1.57_velocity_0.600.pkl'

def visualize_pipeline(N):

    fig, _, axs = utils.subplot2(plt, (2*N+1, 4), (8, 8), (.4, .4))
    axs = axs[::-1]
    filename2 = os.path.join(dirname, filename)
    with open(filename2, 'rb') as f:
        data = pickle.load(f)
   
    lqr_trajectories = Trajectory.init_from_numpy_repr(**data['lqr_trajectories'])
    spline_trajectories = Trajectory.init_from_numpy_repr(**data['spline_trajectories'])
    waypt_configs = SystemConfig.init_from_numpy_repr(**data['waypt_configs'])

    v0 = .6 
    spline_speed_n1 = data['start_speeds']
    lqr_speed_n1 = lqr_trajectories.speed_nk1()[:, 0, :]
    dist = np.abs(v0-lqr_speed_n1)[:, 0]
    idxs = np.argpartition(dist, -N, axis=0)[-N:]

    idxs = [0]
    speeds = spline_trajectories.speed_nk1()[0, :, 0]
    states = forward_sim(speeds.numpy())
    for i, idx in enumerate(idxs):
        axs0, axs1 = axs[2*i*4: (2*i+1)*4], axs[2*i*4+4: (2*i+1)*4+4]
        spline_trajectories.render(axs0, batch_idx=idx, plot_heading=True, plot_velocity=True,
                                label_start_and_end=True, name='Spline')
        lqr_trajectories.render(axs1, batch_idx=idx, plot_heading=True, plot_velocity=True,
                                label_start_and_end=True, name='LQR')
    ax = axs[8]
    ax.plot(states[:, 0], states[:, 1], 'r--')
    fig.savefig('./tmp/problematic_trajectories_dubins_v2.png', bbox_inches='tight')

def forward_sim(velocities):
    states = np.zeros((len(velocities)+1, 2))
    dt = .05
    for i, v in enumerate(velocities):
        states[i+1, 0] = states[i, 0] + v*dt
    return states

if __name__ == '__main__':
    tf.enable_eager_execution()
    visualize_pipeline(1)
    
