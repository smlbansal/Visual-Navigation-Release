from training_utils.trainer_frontend_helper import TrainerFrontendHelper
from utils import utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


class TopViewTrainer(TrainerFrontendHelper):
    """
    Create a trainer that regress on the optimal waypoint using the top-view occupancy maps.
    """

    def parse_params(self, p, args):
        """
        Parse the parameters based on args.command
        to add some additional helpful parameters.
        """
        if args.command == 'generate-data':
            # Change the simulator parameters for data gen if needed
            if hasattr(p.data_creation, 'simulator_params'):
                for key, val in p.data_creation.simulator_params.items():
                    setattr(p.simulator_params, key, val)
        elif args.command == 'train':
            # Change the simulator parameters for training if needed
            if hasattr(p.trainer, 'simulator_params'):
                for key, val in p.trainer.simulator_params.items():
                    setattr(p.simulator_params, key, val)
        elif args.command == 'test':
            # Change the simulator parameters for testing if needed
            if hasattr(p.test, 'simulator_params'):
                for key, val in p.test.simulator_params.items():
                    setattr(p.simulator_params, key, val)
        else:
            raise NotImplementedError('Unknown Command')
        return p

    def create_data_source(self, params=None):
        from data_sources.top_view_trainer_data_source import TopViewDataSource
        self.data_source = TopViewDataSource(self.p)

    def callback_fn(self, epoch):
        """
        A callback function that is called after a training epoch.
        """
        # Instantiate Various Objects Needed for Callbacks
        if epoch == 1:
            self._init_callback_instance_variables() 

        if epoch % self.p.trainer.callback_frequency == 0:
            simulator_data = [{'name': 'NN_Simulator_Epoch_{:d}'.format(epoch),
                               'simulator': self.callback_simulator,
                               'fig': self.callback_fig,
                               'axs': self.callback_axs,
                               'dir': self.callback_dir}]
            metrics_keys, metrics_vals = self.simulate(simulator_data)

            # Log data for visualization via tensorboard
            for k, v in zip(metrics_keys, metrics_vals):
                with self.nn_summary_writer.as_default():
                    with tf.contrib.summary.always_record_summaries():
                        tf.contrib.summary.scalar('metrics/{:s}'.format(k), v, step=epoch)

    def _init_callback_instance_variables(self):
        """Initialize instance variables needed for the callback function."""

        # Initialize the summary writer for tensorboard summaries
        self.nn_summary_writer = tf.contrib.summary.create_file_writer(self._summary_dir(),
                                                                       flush_millis=int(20e3))

        # Parse the simulator params
        self.p.simulator_params.simulator.parse_params(self.p.simulator_params)

        # Instantiate a simulator for callbacks
        nn_simulator_params = self._nn_simulator_params()
        self.callback_simulator = nn_simulator_params.simulator(nn_simulator_params)

        # Instantiate Figure and Axes for plotting
        sqrt_num_plots = int(np.ceil(np.sqrt(self.p.test.number_tests)))
        self.callback_fig, _, self.callback_axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                                                       (8, 8), (.4, .4))
        self.callback_axs = self.callback_axs[::-1]

        # Creates the callback directory
        self.callback_dir = os.path.join(self.p.session_dir, 'callbacks')
        utils.mkdir_if_missing(self.callback_dir)

    def test(self):
        # Call the parent test function first to restore a checkpoint
        super(TopViewTrainer, self).test()

        # Parse the simulator params
        self.p.simulator_params.simulator.parse_params(self.p.simulator_params)

        # The Neural Network Simulator
        nn_simulator_params = self._nn_simulator_params()
        nn_simulator = nn_simulator_params.simulator(nn_simulator_params)

        # Create Figures/ Axes
        sqrt_num_plots = int(np.ceil(np.sqrt(self.p.test.number_tests)))
        fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                     (8, 8), (.4, .4))
        axs = axs[::-1]
        simulator_data = [{'name': 'NN_Simulator',
                           'simulator': nn_simulator,
                           'fig': fig,
                           'axs': axs,
                           'dir': ''}]

        # The Expert Simulator
        if self.p.test.simulate_expert:
            expert_simulator_params = self.p.simulator_params
            expert_simulator = expert_simulator_params.simulator(expert_simulator_params)
            fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                               (8, 8), (.4, .4))
            axs = axs[::-1]
            expert_data = {'name': 'Expert_Simulator',
                           'simulator': expert_simulator,
                           'fig': fig,
                           'axs': axs,
                           'dir': ''}
            simulator_data.append(expert_data)

        self.simulate(simulator_data)

    def simulate(self, simulator_data):
        render_angle_freq = utils.render_angle_frequency(self.p.simulator_params)
        for data in simulator_data:
            name = data['name']
            simulator = data['simulator']
            fig = data['fig']
            axs = data['axs']
            dirname = data['dir']
            metrics = []
            simulator.reset(seed=self.p.test.seed)
            for i in range(self.p.test.number_tests):
                if i != 0:
                    simulator.reset(seed=-1)
                simulator.simulate()
                metrics.append(simulator.get_metrics())
                axs[i].clear()
                simulator.render(axs[i], freq=render_angle_freq)
                axs[i].set_title('#{:d}, {:s}'.format(i, axs[i].get_title()))
            
            metrics_keys, metrics_vals = simulator.collect_metrics(metrics,
                                                                   termination_reasons=self.p.simulator_params.episode_termination_reasons)
            utils.log_dict_as_json(dict(zip(metrics_keys, metrics_vals)),
                                   os.path.join(dirname,
                                                '{:s}.json'.format(name.lower())))
            
            fig.suptitle(name)
            figname = os.path.join(dirname, '{:s}.png'.format(name.lower()))
            fig.savefig(figname, bbox_inches='tight')
            return metrics_keys, metrics_vals

    def _nn_simulator_params(self):
        """
        Returns a DotMap object with simulator parameters
        for a simulator which uses a NN based planner
        """
        from copy import deepcopy
        p = deepcopy(self.p.simulator_params)
        self._modify_planner_params(p)
        return p

    def _modify_planner_params(self, p):
        """
        Modifies a DotMap parameter object
        with parameters for a NNPlanner
        """
        raise NotImplementedError

    def _summary_dir(self):
        """
        Returns the directory name for tensorboard
        summaries
        """
        raise NotImplementedError

if __name__ == '__main__':
    TopViewTrainer().run()
