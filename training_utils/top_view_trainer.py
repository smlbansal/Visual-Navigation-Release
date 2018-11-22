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
            p.simulator_params = p.data_creation.simulator_params
        elif args.command == 'train':
            p.simulator_params = p.trainer.simulator_params
        elif args.command == 'test':
            p.simulator_params = p.test.simulator_params
        else:
            raise NotImplementedError('Unknown Command')

        # Parse the dependencies
        p.simulator_params.simulator.parse_params(p.simulator_params)
        return p

    def create_data_source(self, params=None):
        from data_sources.top_view_trainer_data_source import TopViewDataSource
        self.data_source = TopViewDataSource(self.p)

    def _init_simulator_data(self, p, num_tests, seed, name='', dirname=''):
        """Initializes a simulator_data dictionary based on the params in p,
        num_test, name, and dirname. This can be later passed to the simulate
        function to test a simulator."""
        # Parse the simulator params
        p.simulator.parse_params(p)

        # Initialize the simulator
        simulator = p.simulator(p)

        # Create Figures/ Axes
        sqrt_num_plots = int(np.ceil(np.sqrt(num_tests)))
        fig, _, axs = utils.subplot2(plt, (sqrt_num_plots, sqrt_num_plots),
                                     (8, 8), (.4, .4))
        axs = axs[::-1]

        # Construct data dictionray
        simulator_data = {'name': name,
                          'simulator': simulator,
                          'fig': fig,
                          'axs': axs,
                          'dir': dirname,
                          'n': num_tests,
                          'seed': seed}

        return simulator_data

    def callback_fn(self, lcl):
        """
        A callback function that is called after a training epoch.
        lcl is a key, value mapping of the current state of the local
        variables in the trainer.
        """
        epoch = lcl['epoch']

        # Instantiate Various Objects Needed for Callbacks
        if epoch == 1:
            self._init_callback_instance_variables()

        # Log losses for visualization on tensorboard
        validation_loss = lcl['epoch_performance_validation']
        train_loss = lcl['epoch_performance_training']

        with self.nn_summary_writer.as_default():
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('losses/train', train_loss[-1], step=epoch)
                tf.contrib.summary.scalar('losses/validation', validation_loss[-1], step=epoch)

        if epoch % self.p.trainer.callback_frequency == 0:
            self.simulator_data['name'] = '{:s}_Epoch_{:d}'.format(self.simulator_name,
                                                                   epoch)
            metrics_keyss, metrics_valss = self.simulate([self.simulator_data],
                                                         log_metrics=False)
            metrics_keys = metrics_keyss[0]
            metrics_vals = metrics_valss[0]

            # Log metrics for visualization via tensorboard
            with self.nn_summary_writer.as_default():
                with tf.contrib.summary.always_record_summaries():
                    for k, v in zip(metrics_keys, metrics_vals):
                        tf.contrib.summary.scalar('metrics/{:s}'.format(k.replace(" ", "_")),
                                                  v, step=epoch)

    def _init_callback_instance_variables(self):
        """Initialize instance variables needed for the callback function."""

        # Initialize the summary writer for tensorboard summaries
        self.nn_summary_writer = tf.contrib.summary.create_file_writer(self._summary_dir(),
                                                                       flush_millis=int(20e3))

        # Create the callback directory
        self.callback_dir = os.path.join(self.p.session_dir, 'callbacks')
        utils.mkdir_if_missing(self.callback_dir)

        # Initialize the simulator_data dictionary to be used in callbacks
        nn_simulator_params = self._nn_simulator_params()
        self.simulator_data = self._init_simulator_data(nn_simulator_params,
                                                        self.p.trainer.callback_number_tests,
                                                        self.p.trainer.callback_seed,
                                                        dirname='callbacks')

    def test(self):
        """
        Test a trained network. Optionally test the expert policy as well.
        """
        # Call the parent test function first to restore a checkpoint
        super(TopViewTrainer, self).test()

        with tf.device(self.p.device):
            # Initialize the NN Simulator to be tested
            nn_simulator_params = self._nn_simulator_params()
            nn_simulator_data = self._init_simulator_data(nn_simulator_params,
                                                          self.p.test.number_tests,
                                                          self.p.test.seed,
                                                          name=self.simulator_name)
            simulator_datas = [nn_simulator_data]

            # Optionally initialize the Expert Simulator to be tested
            if self.p.test.simulate_expert:
                expert_simulator_params = self.p.simulator_params
                expert_simulator_data = self._init_simulator_data(expert_simulator_params,
                                                                  self.p.test.number_tests,
                                                                  self.p.test.seed,
                                                                  name='Expert_Simulator')
                simulator_datas.append(expert_simulator_data)

            # Test the simulators
            self.simulate(simulator_datas, log_metrics=True)

    def simulate(self, simulator_datas, log_metrics=True):
        """
        Takes simulator_datas a list of dictionaries of simulator_data. The keys of
        each dictionary are expected to be [name, simulator, fig, axs, dir, n, seed].
        For each simulator, simulates n goals, plots trajectories, and records
        metrics.
        """
        render_angle_freq = utils.render_angle_frequency(self.p.simulator_params)
        metrics_keyss, metrics_valss = [], []
        for data in simulator_datas:
            name = data['name']
            simulator = data['simulator']
            fig = data['fig']
            axs = data['axs']
            dirname = data['dir']
            n = data['n']
            seed = data['seed']
            metrics = []
            simulator.reset(seed=seed)
            for i in range(n):
                if i != 0:
                    simulator.reset(seed=-1)
                simulator.simulate()
                metrics.append(simulator.get_metrics())
                axs[i].clear()
                simulator.render(axs[i], freq=render_angle_freq)
                axs[i].set_title('#{:d}, {:s}'.format(i, axs[i].get_title()))

            # Collect and log the metrics
            metrics_keys, metrics_vals = simulator.collect_metrics(metrics,
                                                                   termination_reasons=self.p.simulator_params.episode_termination_reasons)
            metrics_keyss.append(metrics_keys)
            metrics_valss.append(metrics_vals)
            if log_metrics:
                metrics_filename = os.path.join(self.p.session_dir, dirname,
                                                '{:s}.json'.format(name.lower()))
                utils.log_dict_as_json(dict(zip(metrics_keys, metrics_vals)), metrics_filename)

            # Save the figure
            fig.suptitle(name)
            figname = os.path.join(self.p.session_dir, dirname, '{:s}.png'.format(name.lower()))
            fig.savefig(figname, bbox_inches='tight')
        return metrics_keyss, metrics_valss

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
