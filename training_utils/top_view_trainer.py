from training_utils.trainer_frontend_helper import TrainerFrontendHelper


class TopViewTrainer(TrainerFrontendHelper):
    """
    Create a trainer that regress on the optimal waypoint using the top-view occupancy maps.
    """
    def create_data_source(self, params=None):
        from data_sources.top_view_trainer_data_source import TopViewDataSource
        self.data_source = TopViewDataSource(self.p)

    def test(self):
        super().test()

        # The simulator parameters to simulate the trained neural network
        simulators = [self._test_params()]

        # The simulator parameters to simulate the expert
        if self.p.test.simulate_expert:
            simulators.append(self.p.simulator_params)

        self.simulate(simulators)


    def simulate(simulators):
        # this is generic to be used in the callback function
        # takes a list of simulators and simulates a bunch of goals
        # metrics, etc.
        raise NotImplementedError



if __name__ == '__main__':
    TopViewTrainer().run()
