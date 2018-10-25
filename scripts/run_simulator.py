import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
import argparse

logdir = './logs/simulator'


def simulate(params):
    p = utils.load_params(params)

    print(logdir)
    utils.mkdir_if_missing(logdir)
    utils.log_dict_as_json(p, os.path.join(logdir, 'simulator_params.json'))

    num_tests_per_map = p.control_validation_params.num_tests_per_map
    num_maps = p.control_validation_params.num_maps
    num_plots = num_tests_per_map * num_maps
    sqrt_num_plots = int(np.ceil(np.sqrt(num_plots)))
    fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                 (8, 8), (.4, .4))
    axs = axs[::-1]

    tf.set_random_seed(p.seed)
    np.random.seed(p.seed)
    import pdb; pdb.set_trace()
    sim = p._simulator(params=p)

    k = 0
    metrics = []
    render_angle_freq = int(p.episode_horizon/25)  # heuristic- this looks good
    for i in range(num_maps):
            sim.reset(obstacle_params=obstacle_params)
            for j in range(num_tests_per_map):
                if j != 0:
                    sim.reset()
                sim.simulate()
                metrics.append(sim.get_metrics())
                sim.render(axs[k], freq=render_angle_freq)
                axs[k].set_title('#{:d}, {:s}'.format(k, axs[k].get_title()))
                k += 1
    metrics_keys, metrics_vals = sim.collect_metrics(metrics)
    fig.suptitle('Circular Obstacle Map Simulator')
    figname = os.path.join(logdir, 'circular_obstacle_map.png')
    fig.savefig(figname, bbox_inches='tight')

    utils.log_dict_as_json(dict(zip(metrics_keys, metrics_vals)),
                           os.path.join(logdir, 'metrics.json'))


def main():
    plt.style.use('ggplot')
    tf.enable_eager_execution(config=utils.gpu_config())

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--params', help='parameter version number', default='simulator_params')
    args = parser.parse_args()
    simulate(params=args.params)


if __name__ == '__main__':
        main()
