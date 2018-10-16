import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
import argparse

@profile
def simulate(params):
    """ A script to see how changing dt affects experimental runtime."""
    p = utils.load_params(params)
    print(p.dt)

    num_tests_per_map = p.control_validation_params.num_tests_per_map
    num_maps = p.control_validation_params.num_maps
    num_plots = num_tests_per_map * num_maps
    sqrt_num_plots = int(np.ceil(np.sqrt(num_plots)))
    fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                 (8, 8), (.4, .4))
    axs = axs[::-1]

    tf.set_random_seed(p.seed)
    np.random.seed(p.seed)
    obstacle_params = {'min_n': 4, 'max_n': 7, 'min_r': .3, 'max_r': .8}
    sim = p._simulator(params=p, **p.simulator_params)

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
    fig.savefig('./tmp/change_dt.png', bbox_inches='tight')


def main():
    plt.style.use('ggplot')
    #tf.enable_eager_execution()
    tf.enable_eager_execution(config=utils.gpu_config())

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--params', help='parameter version number', default='v1')
    args = parser.parse_args()
    simulate(params=args.params)


if __name__ == '__main__':
        main()
