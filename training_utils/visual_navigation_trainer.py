from training_utils.trainer_frontend_helper import TrainerFrontendHelper
from utils import utils
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from utils.image_utils import plot_image_observation
import numpy as np
import pickle
import sys


class VisualNavigationTrainer(TrainerFrontendHelper):
    """
    Create a trainer to train a model for navigation using images.
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
        elif args.command in ['test', 'generate-metric-curves']:
            p.simulator_params = p.test.simulator_params
        else:
            raise NotImplementedError('Unknown Command')

        # Parse the dependencies
        p.simulator_params.simulator.parse_params(p.simulator_params)
        return p

    def create_data_source(self, params=None):
        from data_sources.visual_navigation_data_source import VisualNavigationDataSource
        self.data_source = VisualNavigationDataSource(self.p)

        if hasattr(self, 'model'):
            # Give the visual_navigation data source access to the model.
            # May be needed to render training images, etc.
            self.data_source.model = self.model

    def _init_simulator_data(self, p, num_tests, seed, name='', dirname='', plot_controls=False,
                             base_dir=None):
        """Initializes a simulator_data dictionary based on the params in p,
        num_test, name, and dirname. This can be later passed to the simulate
        function to test a simulator."""
        # Parse the simulator params
        p.simulator.parse_params(p)

        # Initialize the simulator
        simulator = p.simulator(p)

        # Create Figures/ Axes
        if plot_controls:
            # Each row has 2 more subplots for linear and angular velocity respectively
            fig, axss, _ = utils.subplot2(plt, (num_tests, 3), (8, 8), (.4, .4))
        else:
            fig, axss, _ = utils.subplot2(plt, (num_tests, 1), (8, 8), (.4, .4))

        if base_dir is None:
            base_dir = self.p.session_dir

        # Construct data dictionray
        simulator_data = {'name': name,
                          'simulator': simulator,
                          'fig': fig,
                          'axss': axss,
                          'dir': dirname,
                          'n': num_tests,
                          'seed': seed,
                          'base_dir': base_dir}

        return simulator_data

    # def callback_fn(self, lcl):
    #     """
    #     A callback function that is called after a training epoch.
    #     lcl is a key, value mapping of the current state of the local
    #     variables in the trainer.
    #     """
    #     epoch = lcl['epoch']
    #
    #     # Instantiate Various Objects Needed for Callbacks
    #     if epoch == 1:
    #         self._init_callback_instance_variables()
    #
    #     # Log losses for visualization on tensorboard
    #     validation_loss = lcl['epoch_performance_validation']
    #     train_loss = lcl['epoch_performance_training']
    #
    #     with self.nn_summary_writer.as_default():
    #         with tf.contrib.summary.always_record_summaries():
    #             tf.contrib.summary.scalar('losses/train', train_loss[-1], step=epoch)
    #             tf.contrib.summary.scalar('losses/validation', validation_loss[-1], step=epoch)
    #
    #     if epoch % self.p.trainer.callback_frequency == 0:
    #         self.simulator_data['name'] = '{:s}_Epoch_{:d}'.format(self.simulator_name,
    #                                                                epoch)
    #         metrics_keyss, metrics_valss = self.simulate([self.simulator_data],
    #                                                      log_metrics=False)
    #         metrics_keys = metrics_keyss[0]
    #         metrics_vals = metrics_valss[0]
    #
    #         # Log metrics for visualization via tensorboard
    #         with self.nn_summary_writer.as_default():
    #             with tf.contrib.summary.always_record_summaries():
    #                 for k, v in zip(metrics_keys, metrics_vals):
    #                     tf.contrib.summary.scalar('metrics/{:s}'.format(k.replace(" ", "_")),
    #                                               v, step=epoch)

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
        super(VisualNavigationTrainer, self).test()

        with tf.device(self.p.device):
            simulator_datas = []

            # If setting custom goals for the agent force the number of tests to be 1
            if self.p.test.simulator_params.reset_params.start_config.position.reset_type == 'custom':
                assert(self.p.test.simulator_params.reset_params.goal_config.position.reset_type == 'custom')
                simulate_kwargs = {}
                number_tests = 1
            else:
                simulate_kwargs = self._ensure_expert_success_data_exists_if_needed()
                number_tests = self.p.test.number_tests

            # Optionally initialize the Expert Simulator to be tested
            if self.p.test.simulate_expert:
                expert_simulator_params = self.p.simulator_params
                expert_simulator_data = self._init_simulator_data(expert_simulator_params,
                                                                  number_tests,
                                                                  self.p.test.seed,
                                                                  name='Expert_Simulator',
                                                                  dirname='expert_simulator',
                                                                  plot_controls=self.p.test.plot_controls)
                simulator_datas.append(expert_simulator_data)

            # Initialize the NN Simulator to be tested
            nn_simulator_params = self._nn_simulator_params()
            nn_simulator_data = self._init_simulator_data(nn_simulator_params,
                                                          number_tests,
                                                          self.p.test.seed,
                                                          name=self.simulator_name,
                                                          dirname=self.simulator_name.lower(),
                                                          plot_controls=self.p.test.plot_controls)
            simulator_datas.append(nn_simulator_data)

            # Test the simulators
            metrics_keys, metrics_values = self.simulate(simulator_datas, log_metrics=True,
                                                         plot_controls=self.p.test.plot_controls,
                                                         plot_images=self.p.test.plot_images,
                                                         **simulate_kwargs)
            return metrics_keys, metrics_values

    def _ensure_expert_success_data_exists_if_needed(self):
        """
        If params.test.use_expert_success_goals is True
        ensure that a file exists with the expert success data.
        If it doesn't, create it.
        """
        if self.p.test.expert_success_goals.use:
            # Check if the expert success goals exist
            expert_dir = os.path.join(self.p.test.expert_success_goals.dirname,
                                      '{:s}_{:s}'.format(self.p.test.simulator_params.obstacle_map_params.renderer_params.dataset_name,
                                                         self.p.test.simulator_params.obstacle_map_params.renderer_params.building_name))
            expert_dir = os.path.join(expert_dir,
                                      '{:d}_goals_{:d}_seed'.format(self.p.test.number_tests,
                                                                    self.p.test.seed))
            # For Python 2.7 Look for py27 pickle compatibles files
            # in the subdirectory py27
            if sys.version_info[0] == 2:
                expert_dir = os.path.join(expert_dir, 'py27')
            expert_success_data_filename = os.path.join(expert_dir, 'expert_success_data.pkl')

            # If the file exists already just load it
            if os.path.exists(expert_success_data_filename):
                with open(expert_success_data_filename, 'rb') as f:
                    data = pickle.load(f)
                expert_success_data = data
            else:
                # Create the expert simulator, run it, and record the metadata
                expert_simulator_params = self.p.simulator_params
                expert_simulator_data = self._init_simulator_data(expert_simulator_params,
                                                                  self.p.test.number_tests,
                                                                  self.p.test.seed,
                                                                  name='Expert_Simulator',
                                                                  dirname='expert_simulator',
                                                                  plot_controls=True,
                                                                  base_dir=expert_dir)
                simulator_datas = [expert_simulator_data]
                metrics_key, metrics_values, episode_types = self.simulate(simulator_datas, log_metrics=True,
                                                                           plot_controls=True, plot_images=True,
                                                                           return_episode_type=True)
                episode_types = np.array(episode_types)
            
                # Save a boolean mask indicating which episodes were invalid
                # Replace these episodes of type -1 with type 0
                invalid_episode_mask = (episode_types == -1)
                episode_types[invalid_episode_mask] = 0

                # Convert episode type numbers to strings
                episode_types_string = np.array(self.p.test.simulator_params.episode_termination_reasons)[episode_types]

                # Replace Invalid episodes with -1 and Invalid respectively
                episode_types_string[invalid_episode_mask] = 'Invalid'
                episode_types[invalid_episode_mask] = -1

                data = {'episode_type_int': episode_types,
                        'episode_types_string': episode_types_string}

                with open(expert_success_data_filename, 'wb') as f:
                    pickle.dump(data, f)

                expert_success_data = data

            # No need to run the expert anymore
            self.p.test.simulate_expert = False

            # Create a boolean valid mask indicating which goals the expert can complete
            kwargs = {'goal_valid_mask': (expert_success_data['episode_types_string'] == 'Success')}
        else:
            kwargs = {}
        return kwargs

    def simulate(self, simulator_datas, log_metrics=True,
                 plot_controls=False, plot_images=False,
                 return_episode_type=False, goal_valid_mask=None):
        """
        Takes simulator_datas a list of dictionaries of simulator_data. The keys of
        each dictionary are expected to be [name, simulator, fig, axs, dir, n, seed].
        For each simulator, simulates n goals, plots trajectories, and records
        metrics.
        """
        metrics_keyss, metrics_valss = [], []
        episode_types = []
        for data in simulator_datas:
            simulator = data['simulator']
            n = data['n']
            seed = data['seed']
            metrics = []
            simulator.reset(seed=seed)

            # goal_valid_mask = None means all goals are valid
            # else goal_valid_mask should be a boolean array of length n
            assert (goal_valid_mask is None or len(goal_valid_mask) == n)
            for i in range(n):
                if i != 0:
                    simulator.reset(seed=-1)

                if goal_valid_mask is None or goal_valid_mask[i]:
                    self._maybe_start_recording_video(i, data)
                    simulator.simulate()
                    self._maybe_stop_recording_video(i, data)
                    if simulator.valid_episode:
                        episode_types.append(simulator.episode_type)
                        metrics.append(simulator.get_metrics())
                        self._plot_episode(i, data, plot_controls=plot_controls,
                                           plot_images=plot_images)
                        self._save_trajectory_data(i, data)
                    else:
                        episode_types.append(-1)

            # Collect and Process the metrics
            metrics_keys, metrics_vals = self._process_metrics(data, metrics, log_metrics)
            metrics_keyss.append(metrics_keys)
            metrics_valss.append(metrics_vals)

            # Save the figure(s)
            self._save_figures(data)
        if return_episode_type:
            return metrics_keyss, metrics_valss, episode_types
        else:
            return metrics_keyss, metrics_valss

    def _process_metrics(self, data, metrics, log_metrics=True):
        simulator = data['simulator']
        name = data['name']
        dirname = data['dir']
        base_dir = data['base_dir']

        # Collect and log the metrics
        metrics_keys, metrics_vals = simulator.collect_metrics(metrics,
                                                               termination_reasons=self.p.simulator_params.episode_termination_reasons)
        if log_metrics:
            metrics_filename = os.path.join(base_dir, dirname,
                                            '{:s}.json'.format(name.lower()))
            utils.log_dict_as_json(dict(zip(metrics_keys, metrics_vals)), metrics_filename)
        return metrics_keys, metrics_vals

    def _plot_episode(self, i, data, plot_controls=False,
                      plot_images=False):
        """
        Render a vehicle trajectory and optionally the associated
        control profiles and associated images.
        """
        render_angle_freq = utils.render_angle_frequency(self.p.simulator_params)

        axss = data['axss']
        simulator = data['simulator']

        prepend_title = '#{:d}, '.format(i)
        axs = axss[i]
        [ax.clear() for ax in axs]
        simulator.render(axs, freq=render_angle_freq, render_velocities=plot_controls,
                         prepend_title=prepend_title)
        
        if plot_images:
            self._plot_episode_images(i, data)
        

    def _save_trajectory_data(self, i, data):
        """
        Optionally log all trajectory data to a pickle file.
        Additionally create a metadata.pkl file which saves
        the episode number and type of every trajectory.
        """
        simulator = data['simulator']
        dirname = data['dir']
        base_dir = data['base_dir']

        if simulator.params.save_trajectory_data:
            trajectory_data_dir = os.path.join(base_dir, dirname, 'trajectories')
            utils.mkdir_if_missing(trajectory_data_dir)

            data = {}
            vehicle_trajectory, vehicle_data, vehicle_data_last_step, vehicle_commanded_actions_1kf = simulator.get_simulator_data_numpy_repr()
            data['vehicle_trajectory'] = vehicle_trajectory
            data['vehicle_data'] = vehicle_data
            data['vehicle_data_last_step'] = vehicle_data_last_step
            data['commanded_actions_1kf'] = vehicle_commanded_actions_1kf

            data['episode_number'] = i
            data['episode_type_int'] = simulator.episode_type
            data['episode_type_string'] = simulator.params.episode_termination_reasons[simulator.episode_type]
            data['valid_episode'] = simulator.valid_episode

            # Current Occupancy Grid- Useful for plotting these trajectories later
            if hasattr(simulator.obstacle_map, 'occupancy_grid_map'):
                data['occupancy_grid'] = simulator.obstacle_map.occupancy_grid_map
                data['map_bounds_extent'] = np.array(simulator.obstacle_map.map_bounds).flatten(order='F')

            trajectory_file = os.path.join(trajectory_data_dir, 'traj_{:d}.pkl'.format(i))
            with open(trajectory_file, 'wb') as f:
                pickle.dump(data, f)

            # Add Trajectory Metadata
            metadata_file = os.path.join(trajectory_data_dir, 'metadata.pkl')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)

                metadata['episode_number'].append(i)
                metadata['episode_type_int'].append(data['episode_type_int'])
                metadata['episode_type_string'].append(data['episode_type_string'])
                metadata['valid_episode'].append(data['valid_episode'])
            else:
                metadata = {}
                metadata['episode_number'] = [i]
                metadata['episode_type_int'] = [data['episode_type_int']]
                metadata['episode_type_string'] = [data['episode_type_string']]
                metadata['valid_episode'] = [data['valid_episode']]

            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)

    def _plot_episode_images(self, i, data):
        """
        Plot the images the robot saw during a particular episode.
        Useful for debugging at test time.
        """
        simulator = data['simulator']
        dirname = data['dir']
        base_dir = data['base_dir']

        imgs_nmkd = simulator.vehicle_data['img_nmkd']
        fig, _, axs = utils.subplot2(plt, (len(imgs_nmkd), 1), (8, 8), (.4, .4))
        axs = axs[::-1]
        for idx, img_mkd in enumerate(imgs_nmkd):
            ax = axs[idx]
            size = img_mkd.shape[0]*simulator.params.obstacle_map_params.dx
            plot_image_observation(ax, img_mkd, size)
            ax.set_title('Img: {:d}'.format(idx))

        figdir = os.path.join(base_dir, dirname, 'imgs')
        utils.mkdir_if_missing(figdir)
        figname = os.path.join(figdir, '{:d}.pdf'.format(i))
        fig.savefig(figname, bbox_inches='tight')
        plt.close(fig)

    def _maybe_start_recording_video(self, i, data):
        """
        If simulator.params.record_video=True then
        call simulator.start_recording_video.
        """
        simulator = data['simulator']

        if simulator.params.record_video:
            simulator.start_recording_video(i)

    def _maybe_stop_recording_video(self, i, data):
        """
        If simulator.params.record_video=True then
        call simulator.stop_recording_video with 
        a file name to save to.
        """
        simulator = data['simulator']
        dirname = data['dir']
        base_dir = data['base_dir']

        if simulator.params.record_video:
            video_dir = os.path.join(base_dir, dirname, 'videos')
            utils.mkdir_if_missing(video_dir)
            video_name = os.path.join(video_dir, '{:d}.mp4'.format(i))
            simulator.stop_recording_video(i, video_name)

    def _save_figures(self, data):
        """
        Save figures with vehicle trajectories and
        optionally control profiles as well.
        """
        fig = data['fig']
        name = data['name']
        dirname = data['dir']
        base_dir = data['base_dir']

        fig.suptitle(name)
        figname = os.path.join(base_dir, dirname, '{:s}.pdf'.format(name.lower()))
        fig.savefig(figname, bbox_inches='tight')

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

    def generate_metric_curves(self):
        """
        Generate a metric curve using a trained network.
        """
        # Extract the number of checkpoints to run
        num_ckpts = self.p.test.metric_curves.end_ckpt - self.p.test.metric_curves.start_ckpt + 1

        # Extract the number of seeds to run
        num_seeds = self.p.test.metric_curves.end_seed - self.p.test.metric_curves.start_seed + 1

        # Checkpoint directory
        ckpt_directory = os.path.join(self.p.trainer.ckpt_path.split('checkpoints')[0], 'checkpoints')

        # Call the test function inside a loop and record the metrics
        for i in range(num_ckpts):
            for j in range(num_seeds):
                # Change the required test and trainer parameters
                self.p.test.seed = j + self.p.test.metric_curves.start_seed
                self.p.test.simulate_expert = False
                self.p.trainer.ckpt_path = os.path.join(ckpt_directory,
                                                        'ckpt-%i' % (i + self.p.test.metric_curves.start_ckpt))

                # Call the test function
                metrics_keys_current, metrics_values_current = self.test()

                # Record the metrics
                if i == 0 and j == 0:
                    # Placeholders for metrics
                    num_metrics = len(metrics_values_current[0])
                    metrics_data = {}
                    metrics_data['values'] = np.zeros((num_seeds, num_ckpts, num_metrics))
                    metrics_data['keys'] = metrics_keys_current[0]
                metrics_data['values'][j, i, :] = metrics_values_current[0]

            self.dump_and_plot_metrics_data(metrics_data)

    def dump_and_plot_metrics_data(self, metrics_data):
        # Dump the metric data
        filename = os.path.join(self.p.session_dir, 'metric_data_ckpts_%i_%i_seeds_%i_%i.pkl' %
                                (self.p.test.metric_curves.start_ckpt, self.p.test.metric_curves.end_ckpt,
                                 self.p.test.metric_curves.start_seed, self.p.test.metric_curves.end_seed))
        with open(filename, 'wb') as handle:
            pickle.dump(metrics_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Plot the metrics
        if self.p.test.metric_curves.plot_curves:
            checkpoints = np.arange(self.p.test.metric_curves.start_ckpt, self.p.test.metric_curves.end_ckpt+1)
            num_metrics = metrics_data['values'].shape[2]
            for i in range(num_metrics):
                mean = np.mean(metrics_data['values'][:, :, i], axis=0)
                std = np.std(metrics_data['values'][:, :, i], axis=0)
                fig = plt.figure(figsize=(8.0, 6.0))
                ax = fig.add_subplot(111)
                ax.plot(checkpoints, mean, 'r-', label=metrics_data['keys'][i])
                ax.fill_between(checkpoints, mean-std, mean+std, color='r', alpha=0.3)
                ax.legend()
                fig.savefig(os.path.join(self.p.session_dir, 'metric_%s.pdf' % metrics_data['keys'][i]))


if __name__ == '__main__':
    VisualNavigationTrainer().run()
