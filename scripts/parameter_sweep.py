import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
import argparse
import os

logdir = './logs'


def parameter_sweep(params):
    p = utils.load_params(params)

    num_tests_per_map = p.control_validation_params.num_tests_per_map
    num_maps = p.control_validation_params.num_maps
    num_plots = num_tests_per_map * num_maps
    sqrt_num_plots = int(np.ceil(np.sqrt(num_plots)))
    fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                 (8, 8), (.4, .4))
    axs = axs[::-1]

    angle_coeffs = np.linspace(0.002, .006, 4)
    collision_coeffs = np.linspace(.5, 2.0, 4)
    goal_coeffs = np.linspace(0.02, .06, 4)

    for angle in angle_coeffs:
        for goal in goal_coeffs:
            for collision in collision_coeffs:
                sim_dir = os.path.join(logdir,
                                       'angle_{:.04f}_goal_{:.04f}_collision_{:.04f}'.format(angle,
                                                                                             goal,
                                                                                             collision))
                p.goal_angle_objective.angle_cost = angle
                p.goal_distance_objective.goal_cost = goal
                p.avoid_obstacle_objective.obstacle_cost = collision
                simulate(p=p, logdir=sim_dir, fig=fig, axs=axs)


def simulate(p, logdir, fig, axs):
    print(logdir)
    utils.mkdir_if_missing(logdir)
    utils.log_dict_as_json(p, os.path.join(logdir, 'params.json'))
    tf.set_random_seed(p.seed)
    np.random.seed(p.seed)
    obstacle_params = {'min_n': 4, 'max_n': 7, 'min_r': .3, 'max_r': .8}
    sim = p._simulator(params=p, **p.simulator_params)

    num_tests_per_map = p.control_validation_params.num_tests_per_map
    num_maps = p.control_validation_params.num_maps

    k = 0
    metrics = []
    for i in range(num_maps):
            sim.reset(obstacle_params=obstacle_params)
            for j in range(num_tests_per_map):
                if j != 0:
                    sim.reset()
                sim.simulate()
                metrics.append(sim.get_metrics())
                sim.render(axs[k], freq=75)
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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--params', help='parameter version number', default='v1')
    args = parser.parse_args()
    parameter_sweep(params=args.params)


if __name__ == '__main__':
        main()
