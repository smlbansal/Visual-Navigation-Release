import numpy as np
from planners.planner import Planner
from trajectory.trajectory import SystemConfig


class NNPlanner(Planner):
    """ A planner which uses
    a trained neural network. """

    def __init__(self, simulator, params):
        super(NNPlanner, self).__init__(simulator, params)
        self.goal_ego_config = SystemConfig(dt=self.params.dt, n=1, k=1)

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        return p

    def _raw_data(self, start_config):
        """
        Return a dictionary of raw_data from the simulator.
        To be passed to model.create_nn_inputs_and_outputs
        """
        simulator = self.simulator
        data = {}

        # Convert Goal to Egocentric Coordinates
        self.params.system_dynamics.to_egocentric_coordinates(start_config,
                                                              simulator.goal_config,
                                                              self.goal_ego_config)

        # Image Data
        if hasattr(self.params.model, 'occupancy_grid_positions_ego_1mk12'):
            kwargs = {'occupancy_grid_positions_ego_1mk12':
                      self.params.model.occupancy_grid_positions_ego_1mk12}
        else:
            kwargs = {}
        data['img_nmkd'] = simulator.get_observation(config=start_config, **kwargs)

        # Vehicle Data
        data['vehicle_state_nk3'] = start_config.position_and_heading_nk3().numpy()
        data['vehicle_controls_nk2'] = start_config.speed_and_angular_speed_nk2().numpy()

        # Goal Data
        data['goal_position_n2'] = simulator.goal_config.position_nk2().numpy()[:, 0, :]
        data['goal_position_ego_n2'] = self.goal_ego_config.position_nk2().numpy()[:, 0, :]

        # Dummy Labels
        data['optimal_waypoint_ego_n3'] = np.ones((1, 3), dtype=np.float32)
        data['waypoint_horizon_n1'] = np.ones((1, 1), dtype=np.float32)
        data['optimal_control_nk2'] = np.ones((1, 1, 2), dtype=np.float32)
        return data
