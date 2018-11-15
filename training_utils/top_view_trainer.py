from training_utils.trainer_frontend_helper import TrainerFrontendHelper
from utils import utils
import numpy as np
import matplotlib.pyplot as plt
import os


class TopViewTrainer(TrainerFrontendHelper):
    """
    Create a trainer that regress on the optimal waypoint using the top-view occupancy maps.
    """
    def create_data_source(self, params=None):
        from data_sources.top_view_trainer_data_source import TopViewDataSource
        self.data_source = TopViewDataSource(self.p)

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
                           'axs': axs}]

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
                           'axs': axs}
            simulator_data.append(expert_data)

        self.simulate(simulator_data)

    def _nn_simulator_params(self):
        """
        Returns a DotMap object with simulator parameters
        for a simulator which uses a NN based planner
        """
        from copy import deepcopy
        p = deepcopy(self.p.simulator_params)
        self._modify_planner_params(p)
        return p

    def simulate(self, simulator_data):
        render_angle_freq = int(self.p.simulator_params.episode_horizon/25)
        for data in simulator_data:
            name = data['name']
            simulator = data['simulator']
            fig = data['fig']
            axs = data['axs']
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
                                   os.path.join(self.p.session_dir,
                                                '{:s}.json'.format(name.lower())))
            
            fig.suptitle(name)
            figname = os.path.join(self.p.session_dir, '{:s}.png'.format(name.lower()))
            fig.savefig(figname, bbox_inches='tight')

    def _planner_params(self):
        """
        Returns a DotMap object with parameters for a
        NN Planner
        """
        raise NotImplementedError


if __name__ == '__main__':
    TopViewTrainer().run()
