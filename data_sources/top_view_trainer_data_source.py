import os
import pickle
import numpy as np
import tensorflow as tf

from data_sources.image_data_source import ImageDataSource
from simulators.circular_obstacle_map_simulator import CircularObstacleMapSimulator
from systems.dubins_car import DubinsCar


class TopViewDataSource(ImageDataSource):

    def _get_n(self, data):
        """
        Returns n, the batch size of the data inside
        this data dictionary.
        """
        return data['vehicle_state_nk3'].shape[0]
    
    # TODO: Varun- look into efficiency at some point to see
    # if data collection can be sped up
    def generate_data(self):
        # Create the data directory if required
        if not os.path.exists(self.p.data_creation.data_dir):
            os.makedirs(self.p.data_creation.data_dir)

        # Initialize the simulator
        simulator = self.p.simulator_params.simulator(self.p.simulator_params)
        
        # Generate the data
        counter = 1
        num_points = 0
        while num_points < self.p.data_creation.data_points:
            # Reset the data dictionary
            data = self.reset_data_dictionary(self.p)
           
            while self._num_data_points(data) < self.p.data_creation.data_points_per_file:
                # Reset the simulator
                simulator.reset()
                
                # Run the planner for one step
                simulator.simulate()

                # Ensure that the episode simulated is valid
                if simulator.valid_episode:
                    # Append the data to the current data dictionary
                    self.append_data_to_dictionary(data, simulator)

            # Prepare the dictionary for saving purposes
            self.prepare_and_save_the_data_dictionary(data, counter)
            
            # Increase the counter
            counter += 1
            num_points += self._num_data_points(data)
            print(num_points)
    
    @staticmethod
    def reset_data_dictionary(params):
        """
        Create a dictionary to store the data.
        """
        # Data dictionary to store the data
        data = {}

        if params.simulator_params.simulator is CircularObstacleMapSimulator:
            # Obstacle information
            data['obs_centers_nm2'] = []
            data['obs_radii_nm1'] = []
        
        # Start configuration information
        data['vehicle_state_nk3'] = []
        data['vehicle_controls_nk2'] = []

        # Goal configuration information
        data['goal_position_n2'] = []
        data['goal_position_ego_n2'] = []

        # Optimal waypoint configuration information
        data['optimal_waypoint_n3'] = []
        data['optimal_waypoint_ego_n3'] = []

        # The horizon of waypoint
        data['waypoint_horizon_n1'] = []

        # Optimal control information
        data['optimal_control_nk2'] = []
        return data

    def _num_data_points(self, data):
        """
        Returns the number of data points inside
        data.
        """
        if type(data['vehicle_state_nk3']) is list:
            if len(data['vehicle_state_nk3']) == 0:
                return 0
            ns = [x.shape[0] for x in data['vehicle_state_nk3']]
            return np.sum(ns)
        elif type(data['vehicle_state_nk3']) is np.ndarray:
            return data['vehicle_state_nk3'].shape[0]
        else:
            raise NotImplementedError

    def append_data_to_dictionary(self, data, simulator):
        """
        Append the appropriate data from the simulator to the existing data dictionary.
        """
        # Batch Dimension
        n = simulator.vehicle_data['system_config'].n

        if self.p.simulator_params.simulator is CircularObstacleMapSimulator:
            # Obstacle data
            obs_center_1m2 = simulator.obstacle_map.obstacle_centers_m2[tf.newaxis, :, :].numpy()
            obs_radii_1m1 = simulator.obstacle_map.obstacle_radii_m1[tf.newaxis, :, :].numpy()

            _, m, _ = obs_center_1m2.shape

            data['obs_centers_nm2'].append(np.broadcast_to(obs_center_1m2, (n, m, 2)))
            data['obs_radii_nm1'].append(np.broadcast_to(obs_radii_1m1, (n, m, 1)))

        # Vehicle data
        data['vehicle_state_nk3'].append(simulator.vehicle_data['trajectory'].position_and_heading_nk3().numpy())
        data['vehicle_controls_nk2'].append(simulator.vehicle_data['trajectory'].speed_and_angular_speed_nk2().numpy())

        # Convert to egocentric coordinates
        start_nk3 = simulator.vehicle_data['system_config'].position_and_heading_nk3().numpy()

        goal_n13 = np.broadcast_to(simulator.goal_config.position_and_heading_nk3().numpy(), (n, 1, 3))
        waypoint_n13 = simulator.vehicle_data['waypoint_config'].position_and_heading_nk3().numpy()

        goal_ego_n13 = DubinsCar.convert_position_and_heading_to_ego_coordinates(start_nk3,
                                                                                 goal_n13)
        waypoint_ego_n13 = DubinsCar.convert_position_and_heading_to_ego_coordinates(start_nk3,
                                                                                     waypoint_n13)

        # Goal Data
        data['goal_position_n2'].append(goal_n13[:, 0, :2])
        data['goal_position_ego_n2'].append(goal_ego_n13[:, 0, :2])

        # Waypoint data
        data['optimal_waypoint_n3'].append(waypoint_n13[:, 0])
        data['optimal_waypoint_ego_n3'].append(waypoint_ego_n13[:, 0])

        # Waypoint horizon
        data['waypoint_horizon_n1'].append(simulator.vehicle_data['planning_horizon_n1'])

        # Optimal control data
        data['optimal_control_nk2'].append(simulator.vehicle_data['trajectory'].speed_and_angular_speed_nk2().numpy())
        return data

    def prepare_and_save_the_data_dictionary(self, data, counter):
        """
        Stack the lists in the dictionary to make an array, and then save the dictionary.
        """
        # Stack the lists
        data_tags = data.keys()
        for tag in data_tags:
            data[tag] = np.concatenate(data[tag], axis=0)

        # Save the data
        filename = os.path.join(self.p.data_creation.data_dir, 'file%i.pkl' % counter)
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
