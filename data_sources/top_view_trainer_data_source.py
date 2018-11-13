import os
import pickle
import numpy as np
import tensorflow as tf

from data_sources.data_source import DataSource
from simulators.circular_obstacle_map_simulator import CircularObstacleMapSimulator
from trajectory.trajectory import Trajectory
from systems.dubins_car import DubinsCar


class TopViewDataSource(DataSource):
   
    # TODO: Varun- look into efficiency at some point to see
    # if data collection can be sped up
    def generate_data(self):
        # Create the data directory if required
        if not os.path.exists(self.p.data_creation.data_dir):
            os.makedirs(self.p.data_creation.data_dir)
        
        # Initialize the simulator
        simulator = CircularObstacleMapSimulator(self.p.simulator_params)
        
        # Initialize the trajectory objects to save the goal and waypoint egocentric configurations
        self.initialize_configs_for_ego_data()
        
        # Generate the data
        counter = 1
        for _ in range(0, self.p.data_creation.data_points, self.p.data_creation.data_points_per_file):
            # Reset the data dictionary
            data = self.reset_data_dictionary()
            
            for _ in range(0, self.p.data_creation.data_points_per_file):
                # Reset the simulator
                simulator.reset()
                
                # Run the planner for one step
                simulator.simulate()
                
                # Append the data to the current data dictionary
                self.append_data_to_dictionary(data, simulator)
                
            # Prepare the dictionary for saving purposes
            self.prepare_and_save_the_data_dictionary(data, counter)
            
            # Increase the counter
            counter += 1
    
    @staticmethod
    def reset_data_dictionary():
        """
        Create a dictionary to store the data.
        """
        # Data dictionary to store the data
        data = {}
        
        # Obstacle information
        data['obs_centers_nm2'] = []
        data['obs_radii_nm1'] = []
        
        # Start configuration information
        data['vehicle_state_n3'] = []
        data['vehicle_controls_n2'] = []

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

    def initialize_configs_for_ego_data(self):
        """
        Creates configuration objects to store the egocentric goal and waypoint.
        """
        self.goal_ego_config = Trajectory(dt=0, n=1, k=1)
        self.waypoint_ego_config = Trajectory(dt=0, n=1, k=1)

    def append_data_to_dictionary(self, data, simulator):
        """
        Append the appropriate data from the simulator to the existing data dictionary.
        """
        # Convert the waypoint and the goal information to egocentric frame
        DubinsCar.to_egocentric_coordinates(simulator.start_config, simulator.waypt_configs[0],
                                            self.waypoint_ego_config)
        DubinsCar.to_egocentric_coordinates(simulator.start_config, simulator.goal_config, self.goal_ego_config)

        # Obstacle data
        data['obs_centers_nm2'].append(simulator.obstacle_map.obstacle_centers_m2[tf.newaxis, :, :].numpy())
        data['obs_radii_nm1'].append(simulator.obstacle_map.obstacle_radii_m1[tf.newaxis, :, :].numpy())

        # Vehicle data
        data['vehicle_state_n3'].append(simulator.start_config.position_and_heading_nk3().numpy()[:, 0, :])
        data['vehicle_controls_n2'].append(simulator.start_config.speed_and_angular_speed_nk2().numpy()[:, 0, :])

        # Goal data
        data['goal_position_n2'].append(simulator.goal_config.position_nk2().numpy()[:, 0, :])
        data['goal_position_ego_n2'].append(self.goal_ego_config.position_nk2().numpy()[:, 0, :])

        # Waypoint data
        data['optimal_waypoint_n3'].append(simulator.waypt_configs[0].position_and_heading_nk3().numpy()[:, 0, :])
        data['optimal_waypoint_ego_n3'].append(self.waypoint_ego_config.position_and_heading_nk3().numpy()[:, 0, :])

        # Waypoint horizon
        data['waypoint_horizon_n1'].append(simulator.waypt_horizons[0])

        # Optimal control data
        data['optimal_control_nk2'].append(simulator.vehicle_trajectory.speed_and_angular_speed_nk2().numpy())
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
