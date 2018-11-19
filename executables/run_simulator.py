import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
import pickle
import argparse

logdir = './logs/simulator'

# TODO: the vehicle_trajectory is not exactly the LQR reference trajectory.
# It is very close though so this may not really be problematic.

#TODO: This wont work anymore as simulator has changed
def save_lqr_data(filename, trajectory, controllers):
        """ Saves the LQR controllers (K, k) used to track the current vehicle
        trajectory as well as the current vehicle trajectory."""
        data = {'trajectory' : trajectory.to_numpy_repr(),
                'K_1kfd' : controllers['K_1kfd'].numpy(),
                'k_1kf1' : controllers['k_1kf1'].numpy()}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


def plot_velocity_profile_hisotogram(v, w):
    data_dir = os.path.join('./tmp', 'expert_data_distribution') 
    utils.mkdir_if_missing(data_dir)

    v_1k1 = np.concatenate(v, axis=1)
    w_1k1 = np.concatenate(w, axis=1)

    data = {'v': v_1k1,
            'w': w_1k1}
    filename = os.path.join(data_dir, 'data.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    fig, _, axs = utils.subplot2(plt, (1, 2), (8, 8), (.4, .4))
    axs = axs[::-1]
    
    ax = axs[0]
    ax.hist(v_1k1[0, :, 0], bins=61, range=(0.0, .6), density=True)
    ax.set_title('Velocity Histogram')

    ax = axs[1]
    ax.hist(w_1k1[0, :, 0], bins=241, range=(-1.2, 1.2), density=True)
    ax.set_title('Omega Histogram')

    fig.savefig(os.path.join(data_dir, 'velocity_profiles.png'), bbox_inches='tight')


v = []
w = []


def simulate(plot_controls=False):
    p = utils.load_params('simulator_params')
    print(logdir)
    utils.mkdir_if_missing(logdir)
    utils.log_dict_as_json(p, os.path.join(logdir, 'simulator_params.json'))

    sqrt_num_plots = int(np.ceil(np.sqrt(p.num_validation_goals)))
    fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                 (8, 8), (.4, .4))
    axs = axs[::-1]
    if plot_controls:
        fig0, _, axs0 = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                       (8, 8), (.4, .4))
        fig1, _, axs1 = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                       (8, 8), (.4, .4))
        axs0 = axs0[::-1]
        axs1 = axs1[::-1]

    tf.set_random_seed(p.seed)
    np.random.seed(p.seed)

    sim = p.simulator(params=p)

    metrics = []
    # heuristic- this looks good
    render_angle_freq = int(p.episode_horizon / 25)
    sim.reset(seed=p.seed)
    for i in range(p.num_validation_goals):
        print(i)
        if i != 0:
            sim.reset(seed=-1)
        sim.simulate()
        metrics.append(sim.get_metrics())

        ### Debugging stuff
        v.append(sim.vehicle_trajectory.speed_nk1().numpy())
        w.append(sim.vehicle_trajectory.angular_speed_nk1().numpy())

        # Plot Stuff
        axs[i].clear()
        sim.render(axs[i], freq=render_angle_freq)
        axs[i].set_title('#{:d}, {:s}'.format(i, axs[i].get_title()))

        if plot_controls:
            axs0[i].clear()
            axs1[i].clear()
            sim.render_velocities(axs0[i], axs1[i])
            axs0[i].set_title('#{:d}, {:s}'.format(i, axs0[i].get_title()))
            axs1[i].set_title('#{:d}, {:s}'.format(i, axs1[i].get_title()))

    
    plot_velocity_profile_hisotogram(v, w)
    metrics_keys, metrics_vals = sim.collect_metrics(metrics,
                                                     termination_reasons=p.simulator_params.episode_termination_reasons)
    fig.suptitle('Circular Obstacle Map Simulator')
    figname = os.path.join(logdir, 'circular_obstacle_map.png')
    fig.savefig(figname, bbox_inches='tight')

    if plot_controls:
        fig0.suptitle('Circular Obstacle Map Simulator Velocity Profile')
        figname = os.path.join(
            logdir, 'circular_obstacle_map_velocity_profile.png')
        fig0.savefig(figname, bbox_inches='tight')

        fig1.suptitle('Circular Obstacle Map Simulator Omega Profile')
        figname = os.path.join(
            logdir, 'circular_obstacle_map_omega_profile.png')
        fig1.savefig(figname, bbox_inches='tight')

    utils.log_dict_as_json(dict(zip(metrics_keys, metrics_vals)),
                           os.path.join(logdir, 'metrics.json'))


def main():
    plt.style.use('ggplot')
    tf.enable_eager_execution(**utils.tf_session_config())

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plot_controls', type=bool, default=False)
    args = parser.parse_args()
    simulate(plot_controls=args.plot_controls)


if __name__ == '__main__':
    main()
