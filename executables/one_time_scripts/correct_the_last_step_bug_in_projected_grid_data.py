import pickle
import numpy as np
import os

trajectory_segments_per_episode = 14
file_folder = '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid'
area_name = 'area5a'
num_files = 70
new_file_folder = '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid_corrected_last_step'


def load_and_correct_pickle_files():
    # Old and new data directories
    old_dir = os.path.join(file_folder, area_name, 'full_episode_random_v1_100k')
    new_dir = os.path.join(new_file_folder, area_name, 'full_episode_random_v1_100k')
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    for j in range(num_files):
    
        # Find the filename
        filename = os.path.join(old_dir, 'file%i.pkl' % (j+1))
    
        # Load the file
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        
        # Let's correct the file
        num_episodes = data['episode_number_n1'][-1] - data['episode_number_n1'][0] + 1
    
        # Arrays to store the corrected last episode data
        corrected_data = initialize_the_corrected_data_dictionary()
        last_step_data_keys = corrected_data.keys()
        
        # Relevant index of the last step data
        start_index = 0
    
        for i in range(num_episodes):
            # Find the episode indices
            episode_indices = np.where(data['episode_number_n1'] == i)[0]
    
            # Remove the last segments as required
            if data['episode_type_string_n1'][episode_indices[0]] == 'Timeout':
                assert episode_indices.shape[0] == (trajectory_segments_per_episode - 1)
                # Fetch the correct data
                corrected_data = fetch_the_correct_last_step_data(data, corrected_data, start_index)
                # Increment the relevant index in the last step data
                start_index = start_index + 1
            elif data['episode_type_string_n1'][episode_indices[0]] == 'Success':
                # Fetch the correct last step data
                corrected_data = fetch_the_correct_last_step_data(data, corrected_data, start_index)
                # Increment the relevant index in the last step data
                start_index = start_index + trajectory_segments_per_episode - episode_indices.shape[0]
            else:
                raise NotImplementedError
        
        # Do some assertion checks
        assert start_index == data['last_step_vehicle_state_nk3'].shape[0]
            
        # Assign the corrected data to the data
        data = assign_the_corrected_data(data, corrected_data)
        
        # Do some final assertion checks before saving
        for key in last_step_data_keys:
            assert corrected_data[key].shape[0] == num_episodes
            assert data[key].shape[0] == num_episodes
        
        # Save the file
        filename = os.path.join(new_dir, 'file%i.pkl' % (j + 1))
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        
def initialize_the_corrected_data_dictionary():
    corrected_data = {}
    corrected_data['last_step_vehicle_state_nk3'] = []
    corrected_data['last_step_vehicle_controls_nk2'] = []
    corrected_data['last_step_goal_position_n2'] = []
    corrected_data['last_step_goal_position_ego_n2'] = []
    corrected_data['last_step_optimal_waypoint_n3'] = []
    corrected_data['last_step_optimal_waypoint_ego_n3'] = []
    corrected_data['last_step_optimal_control_nk2'] = []
    return corrected_data
    
    
def fetch_the_correct_last_step_data(data, corrected_data, start_index):
    relevant_keys = corrected_data.keys()
    for key in relevant_keys:
        corrected_data[key].append(data[key][start_index:start_index+1])
    return corrected_data


def assign_the_corrected_data(data, corrected_data):
    relevant_keys = corrected_data.keys()
    for key in relevant_keys:
        corrected_data[key] = np.concatenate(corrected_data[key], axis=0)
        data[key] = corrected_data[key] * 1.
    return data
    

if __name__ == '__main__':
    load_and_correct_pickle_files()
