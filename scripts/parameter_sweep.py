import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
import argparse
import logging
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

    angle_coeffs = np.linspace(0.0, 1.0, 6)
    collision_coeffs = np.linspace(0.0, 1.0, 6)
    goal_coeffs = np.linspace(0.0, 1.0, 6)
 
    for angle in angle_coeffs:
        for goal in goal_coeffs:
            for collision in collision_coeffs:
                sim_dir = os.path.join(logdir,
                                       'angle_{:.02f}_goal_{:.02f}_collision_{:.02f}'.format(angle,
                                                                                             goal,
                                                                                             collision))
                p.goal_angle_objective.angle_cost = angle
                p.goal_distance_objective.goal_cost = goal
                p.avoid_obstacle_objective.obstacle_cost = collision
                simulate(p=p, logdir=sim_dir, fig=fig, axs=axs)


def simulate(p, logdir, fig, axs):
    utils.mkdir_if_missing(logdir)
    print(logdir)
    logging.basicConfig(filename=os.path.join(logdir, 'log.log'), filemode='w')

    utils.log_params(p, os.path.join(logdir, 'params.json'))
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
                k += 1
                if j != 0:
                    sim.reset()
                sim.simulate()
                metrics.append(sim.get_metrics())
                sim.render(axs[k], freq=4)
    metrics_keys, metrics_vals = sim.collect_metrics(metrics)
    fig.suptitle('Circular Obstacle Map Simulator')
    figname = os.path.join(logdir, 'circular_obstacle_map.png')
    fig.savefig(figname, bbox_inches='tight')

    for key, val in zip(metrics_keys, metrics_vals):
        logging.error('{:s}: {:f}'.format(key, val))


def main():
    plt.style.use('ggplot')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--params', help='parameter version number', default='v1')
    args = parser.parse_args()
    parameter_sweep(params=args.params)


if __name__ == '__main__':
        main()
