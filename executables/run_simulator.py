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

    sqrt_num_plots = int(np.ceil(np.sqrt(p.num_validation_goals)))
    fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                 (8, 8), (.4, .4))
    axs = axs[::-1]

    tf.set_random_seed(p.seed)
    np.random.seed(p.seed)

    sim = p._simulator(params=p)

    metrics = []
    render_angle_freq = int(p.episode_horizon/25)  # heuristic- this looks good
    sim.reset(seed=p.seed)
    for i in range(p.num_validation_goals):
        print(i)
        if i != 0:
            sim.reset(seed=-1)
        if i != 1:
            continue
        sim.simulate()
        metrics.append(sim.get_metrics())
        axs[i].clear()
        sim.render(axs[i], freq=render_angle_freq)
        axs[i].set_title('#{:d}, {:s}'.format(i, axs[i].get_title()))
    metrics_keys, metrics_vals = sim.collect_metrics(metrics,
                                                     termination_reasons=p.simulator_params.episode_termination_reasons)
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
