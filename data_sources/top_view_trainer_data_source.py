import os
import pickle
import numpy as np

from data_sources.data_source import DataSource
from simulators.circular_obstacle_map_simulator import CircularObstacleMapSimulator


class TopViewDataSource(DataSource):
    
    def generate_data(self):
        # Create the data directory if required
        if not os.path.exists(self.p.data_creation.data_dir):
            os.makedirs(self.p.data_creation.data_dir)
        
        # Initialize the simulator
        simulator = CircularObstacleMapSimulator(self.p.simulator_params)
        
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
    
    def reset_data_dictionary(self):
        """
        Create a dictionary to store the data.
        """
        data = {}
        data['obs_centers_n2'] = []
        data['obs_radii_n2'] = []
        data['vehicle_state_n3'] = []
        data['vehicle_controls_n2'] = []
        data['goal_position_n2'] = []
        data['optimal_waypoint_n3'] = []
        # TODO(Varun): Add the logic in simulator to fetch the horizon and then add the logic here to fetch the data.
        # data['waypoint_horizon_n1'] = []
        data['optimal_control_nk2'] = []
        return data
    
    def append_data_to_dictionary(self, data, simulator):
        """
        Append the appropriate data from the simulator to the existing data dictionary.
        """
        # Obstacle data
        data['obs_centers_n2'].append(simulator.obstacle_map.obstacle_centers_m2.numpy())
        data['obs_radii_n2'].append(simulator.obstacle_map.obstacle_radii_m1.numpy())
        
        # Vehicle data
        data['vehicle_state_n3'].append(simulator.start_config.position_and_heading_nk3().numpy()[:, 0, :])
        data['vehicle_controls_n2'].append(simulator.start_config.speed_and_angular_speed_nk2().numpy()[:, 0, :])
        
        # Goal data
        data['goal_position_n2'].append(simulator.goal_config.position_nk2().numpy()[:, 0, :])
        
        # Waypoint data
        data['optimal_waypoint_n3'].append(simulator.waypt_configs[0].position_and_heading_nk3().numpy()[:, 0, :])
        # TODO(Varun): Add the functionality of saving the horizon.
        
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
            data[tag] = np.stack(data[tag], axis=0)

        # Save the data
        filename = os.path.join(self.p.data_creation.data_dir, 'file%i.pkl' % counter)
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
