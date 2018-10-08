import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
import argparse


def simulate(params):
    p = utils.load_params(params)
    tf.set_random_seed(p.seed)
    np.random.seed(p.seed)
    obstacle_params = {'min_n': 2, 'max_n': 3, 'min_r': .15, 'max_r': .5}
    sim = p._simulator(params=p, **p.simulator_params)

    num_tests_per_map = p.control_validation_params.num_tests_per_map
    num_maps = p.control_validation_params.num_maps
    num_plots = num_tests_per_map * num_maps
    sqrt_num_plots = int(np.ceil(np.sqrt(num_plots)))
    fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                 (8, 8), (.4, .4))
    axs = axs[::-1]
    k = 0
    metrics = []
    for i in range(num_maps):
            sim.reset(obstacle_params=obstacle_params)
            for j in range(num_tests_per_map):
                print(k)
                k += 1
                if j != 0:
                    sim.reset()
                sim.simulate()
                metrics.append(sim.get_metrics())
                ax = axs.pop()
                sim.render(ax, freq=4)
    metrics_keys, metrics_vals = sim.collect_metrics(metrics)
    fig.suptitle('Circular Obstacle Map Simulator')
    fig.savefig('./tmp/circular_obstacle_map.png', bbox_inches='tight')

    for key, val in zip(metrics_keys, metrics_vals):
        print('{:s}: {:f}'.format(key, val))


def main():
    plt.style.use('ggplot')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--params', help='parameter version number', default='v1')
    args = parser.parse_args()
    simulate(params=args.params)


if __name__ == '__main__':
        main()
