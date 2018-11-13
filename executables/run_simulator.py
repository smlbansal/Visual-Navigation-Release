import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
import pickle
import argparse

logdir = './logs/simulator'


def save_lqr_data(filename, trajectory, controllers):
        """ Saves the LQR controllers (K, k) used to track the current vehicle
        trajectory as well as the current vehicle trajectory."""
        data = {'trajectory' : trajectory.to_numpy_repr(),
                'K_1kfd' : controllers['K_1kfd'].numpy(),
                'k_1kf1' : controllers['k_1kf1'].numpy()}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


def simulate(plot_controls=False):
    p = utils.load_params('simulator_params')
    print(logdir)
    utils.mkdir_if_missing(logdir)
    utils.mkdir_if_missing(os.path.join(logdir, 'lqr_data'))
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

    tf.set_random_seed(p.common.seed)
    np.random.seed(p.common.seed)

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
        lqr_filename = os.path.join(logdir, 'lqr_data', 'goal_num_{:d}.pkl'.format(i))
        save_lqr_data(lqr_filename, sim.vehicle_trajectory, sim.controllers)
        metrics.append(sim.get_metrics())

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
    tf.enable_eager_execution(config=utils.gpu_config())

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plot_controls', type=bool, default=False)
    args = parser.parse_args()
    simulate(plot_controls=args.plot_controls)


if __name__ == '__main__':
    main()
