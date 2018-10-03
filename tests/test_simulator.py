import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
from simulators.circular_obstacle_map_simulator import CircularObstacleMapSimulator
from utils import utils

def simulate(params):
    num_tests_per_map = 1
    num_maps = 4
    num_plots = num_tests_per_map * num_maps

    p = utils.load_params()
    tf.set_random_seed(p.seed)
    np.random.seed(p.seed)
    obstacle_params = {'min_n': 2, 'max_n': 3, 'min_r': .15, 'max_r': .5}
    sim = p._simulator(params=p, **p.simulator_params)

    sqrt_num_plots = int(np.ceil(np.sqrt(num_plots)))
    fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots), (8, 8), (.4, .4))
    axs = axs[::-1]
    for i in range(num_maps):
            sim.reset(obstacle_params=obstacle_params)
            for j in range(num_tests_per_map):
                if j != 0:
                    sim.reset()
                sim.simulate()
                ax = axs.pop()
                sim.render(ax, freq=4)
    fig.suptitle('Circular Obstacle Map Simulator')
    plt.show()


def main():
    plt.style.use('ggplot')
    simulate('v0')


if __name__ == '__main__':
    main()
