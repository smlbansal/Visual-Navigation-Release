import tensorflow as tf
from planners.planner import Planner
from trajectory.trajectory import SystemConfig


class NNPlanner(Planner):
    """ A planner which uses
    a trained neural network. """

    def __init__(self, simulator, params):
        super().__init__(simulator, params)
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
        self.params.system_dynamics.to_egocentric_coordinates(start_config, simulator.goal_config, self.goal_ego_config)
        
        # Obstacle Data
        data['obs_centers_nm2'] = simulator.obstacle_map.obstacle_centers_m2[tf.newaxis, :, :].numpy()
        data['obs_radii_nm1'] = simulator.obstacle_map.obstacle_radii_m1[tf.newaxis, :, :].numpy()

        # Vehicle Data
        data['vehicle_state_n3'] = start_config.position_and_heading_nk3().numpy()[:, 0, :]
        data['vehicle_controls_n2'] = start_config.speed_and_angular_speed_nk2().numpy()[:, 0, :]

        # Goal Data
        data['goal_position_n2'] = simulator.goal_config.position_nk2().numpy()[:, 0, :]
        data['goal_position_ego_n2'] = self.goal_ego_config.position_nk2().numpy()[:, 0, :]

        # Label Data
        data['optimal_waypoint_ego_n3'] = None
        data['waypoint_horizon_n1'] = None
        data['optimal_control_nk2'] = None
        return data
