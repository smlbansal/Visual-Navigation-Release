import os
import pickle
import numpy as np
import tensorflow as tf

from data_sources.data_source import DataSource
from simulators.circular_obstacle_map_simulator import CircularObstacleMapSimulator
from trajectory.trajectory import SystemConfig
from systems.dubins_car import DubinsCar


class TopViewDataSource(DataSource):
    
    def _prepare_for_data_loading(self):
        from utils import utils 
        import datetime

        # Create a temporary directory for image data
        img_dir = 'tmp_{:s}_image_data_{:s}'.format(self.p.simulator_params.obstacle_map_params.renderer_params.camera_params.modalities[0],
                                                    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.img_dir = os.path.join(self.p.data_creation.data_dir, img_dir)
        utils.mkdir_if_missing(self.img_dir)

        # Initialize the simulator to render images
        simulator = self.p.simulator_params.simulator(self.p.simulator_params)


        data_files = self.get_file_list()
        for data_file in data_files:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
        
            # data_file should be ../path/file{:d}.pkl
            filename = data_file.split('/')[-1]  # file{:d}.pkl
            file_number = filename.split('.')[0]  # file{:d}
            file_number = int(file_number.split('file')[-1])  # {:d}
           
            # Add the tag 'img_placeholder_n2' to data if needed
            if not 'img_placeholder_n2' in data.keys():
                n = data['vehicle_state_nk3'].shape[0]
                data['img_placeholder_n2'] = np.stack([np.ones(n, dtype=np.int32)*file_number,
                                                       np.arange(n)], axis=1)
                # Save the data back to the data_file
                with open(data_file, 'wb') as f:
                    pickle.dump(data, f)

            # Render the images from the simulator
            if simulator.name == 'Circular_Obstacle_Map_Simulator':
                import pdb; pdb.set_trace()
                # TODO: get the occupancy grid from somewhere!!!
                img_nmkd = simulator.get_observation(pos_n3=data['vehicle_state_nk3'][:, 0],
                                                     obs_centers_nl2=data['obs_centers_nm2'],
                                                     obs_radii_nl1=data['obs_radii_nm1'],
                                                     occupancy_grid_positions_ego_1mk12=self.occupancy_grid_positions_ego_1mk12)
            elif simulator.name == 'SBPD_Simulator':
                img_nmkd = simulator.get_observation(pos_n3=data['vehicle_state_nk3'][:, 0],
                                                     crop_size=self.p.model.num_inputs.occupancy_grid_size)
            else:
                raise NotImplementedError
            
            # Save the images to the img dir
            img_filename = os.path.join(self.img_dir, filename)
            data = {'img_nmkd': np.array(img_nmkd)}
            with open(img_filename, 'wb') as f:
                pickle.dump(data, f)
            #TODO: Delete the img dir later

    def generate_training_batch(self, start_index):
        """
        Generate a training batch from the dataset.
        """
        data = super().generate_training_batch(start_index)
        self._load_images_into_data(data)
        return data
            
    def generate_validation_batch(self):
        """
        Generate a validation batch from the dataset.
        """
        data = super().generate_validation_batch()
        self._load_images_into_data(data)
        return data

    def _load_images_into_data(self, data):
        """
        Use the information stored in
        data['img_placeholder_n2'] to load
        the images for training.
        """
        img_placeholder_n2 = data['img_placeholder_n2']
        file_numbers = set(img_placeholder_n2[:, 0])
        img_nmkd = np.zeros((img_placeholder_n2.shape[0],
                             *self.p.model.num_inputs.occupancy_grid_size), dtype=np.float32)
        for file_number in file_numbers:
            filename = os.path.join(self.img_dir, 'file{:d}.pkl'.format(file_number))
            with open(filename, 'rb') as f:
                img_data = pickle.load(f)

            batch_mask = (img_placeholder_n2[:, 0] == file_number)
            data_idxs = img_placeholder_n2[:, 1][batch_mask]
            #TODO: Check if adding or equality is faster here
            img_nmkd[batch_mask] += img_data['img_nmkd'][data_idxs]

        data['img_nmkd'] = img_nmkd

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
