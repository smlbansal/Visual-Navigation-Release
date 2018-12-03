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

#TODO: This will no longer work
def log_images(i, sim, logdir):
    logdir = os.path.join(logdir, str(i))
    utils.mkdir_if_missing(logdir)
    imgs_nmkd = sim.get_observation(sim.vehicle_data['system_config'])  
    fig, _, axs = utils.subplot2(plt, (len(imgs), 1), (8, 8), (.4, .4)) 
    axs = axs[::-1]
    for idx, img_mkd in enumerate(imgs_nmkd):
        ax = axs[idx]
        if img_mkd.shape[2] == 1:  # plot a topview image
            size = img_mkd.shape[0]*sim.params.obstacle_map_params.dx
            ax.imshow(img_mkd[:, :, 0], cmap='gray', extent=(0, size, -size/2.0, size/2.0))

            # Plot the robot position and heading for convenience
            ax.plot(0, 0, 'r.', markersize=20)
            ax.quiver(0, 0, 1., 0., color='red')

        else:
            ax.imshow(img_mkd.astype(np.int32))
        ax.set_title('Image {:d}'.format(idx))
        ax.grid('off')
    filename = os.path.join(logdir, 'fpv.png')
    fig.savefig(filename, bbox_inches='tight')

v = []
w = []


def simulate(plot_controls, log_fpv_images, save_lqr, plot_velocity_hist):
    p = utils.load_params('simulator_params')
    print(logdir)
    utils.mkdir_if_missing(logdir)
    utils.log_dict_as_json(p, os.path.join(logdir, 'simulator_params.json'))

    if plot_controls:
        fig, axss, _ = utils.subplot2(plt, (p.num_validation_goals, 3),
                                      (8, 8), (.4, .4))
    else:
        fig, axss, _ = utils.subplot2(plt, (p.num_validation_goals, 1),
                                      (8, 8), (.4, .4))

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
        if sim.valid_episode:
            metrics.append(sim.get_metrics())

            # Plot Stuff
            prepend_title = '#{:d}, '.format(i)
            axs = axss[i]
            sim.render(axs, freq=render_angle_freq, render_velocities=plot_controls,
                       prepend_title=prepend_title)
       
            if log_fpv_images:
                log_images(i, sim, logdir)

            if plot_velocity_hist:
                v.append(sim.vehicle_trajectory.speed_nk1().numpy())
                w.append(sim.vehicle_trajectory.angular_speed_nk1().numpy())

    metrics_keys, metrics_vals = sim.collect_metrics(metrics,
                                                     termination_reasons=p.simulator_params.episode_termination_reasons)
    fig.suptitle('{:s}'.format(sim.name))
    figname = os.path.join(logdir, '{:s}.pdf'.format(sim.name.lower()))
    fig.savefig(figname, bbox_inches='tight', pad_inches=0)

    utils.log_dict_as_json(dict(zip(metrics_keys, metrics_vals)),
                           os.path.join(logdir, 'metrics.json'))
    if plot_velocity_hist:
        plot_velocity_profile_hisotogram(v, w)


def main():
    plt.style.use('ggplot')
    tf.enable_eager_execution(**utils.tf_session_config())

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plot_controls', type=bool, default=False)
    parser.add_argument('--log_fpv_images', type=bool, default=False)
    parser.add_argument('--save_lqr', type=bool, default=False)
    parser.add_argument('--plot_velocity_hist', type=bool, default=False)
    args = parser.parse_args()
    simulate(plot_controls=args.plot_controls,
             log_fpv_images=args.log_fpv_images,
             save_lqr=args.save_lqr,
             plot_velocity_hist=args.plot_velocity_hist
             )


if __name__ == '__main__':
    main()
