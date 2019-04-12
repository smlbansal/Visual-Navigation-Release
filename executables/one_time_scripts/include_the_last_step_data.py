import pickle
import numpy as np
import os

trajectory_segments_per_episode = 40
file_folder = '/home/ext_drive/somilb/data/training_data/sbpd/max_fmm_dist_20/sbpd_projected_grid'
area_name = 'area4'
num_files = 99
new_file_folder = '/home/ext_drive/somilb/data/training_data/sbpd/max_fmm_dist_20/sbpd_projected_grid_include_last_step_successful_goals_only'

# Store metadata
store_metadata = True
metadata_prefix = 'img_data_rgb_1024_1024_3_90.00_90.00_0.01_20.00_0.22_18_10_100_80_-45_1.000'

# Only keep successful goals
filter_timeout_episodes = True


def load_and_correct_pickle_files():
    
    if store_metadata:
        metadata = {}
    
    # Old and new data directories
    old_dir = os.path.join(file_folder, area_name, 'full_episode_random_v1_100k')
    new_dir = os.path.join(new_file_folder, area_name, 'full_episode_random_v1_100k')
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    # Last step keys
    last_step_data_keys = ['last_step_vehicle_state_nk3', 'last_step_vehicle_controls_nk2',
                           'last_step_goal_position_n2', 'last_step_goal_position_ego_n2',
                           'last_step_optimal_waypoint_n3', 'last_step_optimal_waypoint_ego_n3',
                           'last_step_optimal_control_nk2']
    
    other_data_keys = ['vehicle_state_nk3', 'vehicle_controls_nk2',
                       'goal_position_n2', 'goal_position_ego_n2',
                       'optimal_waypoint_n3', 'optimal_waypoint_ego_n3', 'optimal_control_nk2']
    
    for j in range(num_files):
    
        # Find the filename
        filename = os.path.join(old_dir, 'file%i.pkl' % (j+1))
    
        # Load the file
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        
        if filter_timeout_episodes:
            # Find the successful episodes
            indices_for_successful_episodes = np.where(data['episode_type_string_n1'] == 'Success')[0]
            successful_episodes_numbers = np.unique(data['episode_number_n1'][indices_for_successful_episodes])
          
            # Only keep the data corresponding to the successful episodes in the main fields
            data = keep_the_successful_episodes_data(data, other_data_keys, indices_for_successful_episodes)
            
            # Only keep the data corresponding to the successful episodes in the last trajectory segment fields
            data = keep_the_successful_episodes_data(data, last_step_data_keys, successful_episodes_numbers)
        
            # Append the last segment data
            data = append_the_last_segment_data_and_delete_last_step_keys(data, other_data_keys)
            
            # Do some assertion checks
            indices_for_timeout_episodes = np.where(data['episode_type_string_n1'] == 'Timeout')[0]
            indices_for_collision_episodes = np.where(data['episode_type_string_n1'] == 'Collision')[0]

            timeout_episodes_numbers = np.unique(data['episode_number_n1'][indices_for_timeout_episodes])

            # There may be a couple of episodes with collisions
            # now that we sample problems with larger distances (up to 20 meters)
            number_of_successful_episodes = successful_episodes_numbers.shape[0]
            number_of_timeout_episodes = timeout_episodes_numbers.shape[0]
            num_collision_data_pts = indices_for_collision_episodes.shape[0]
            
            assert data['vehicle_state_nk3'].shape[0] == \
                   data['episode_number_n1'].shape[0] - number_of_timeout_episodes * \
                   (trajectory_segments_per_episode-1) + \
                   number_of_successful_episodes - num_collision_data_pts
        else:
            # Just append the last segment data
            data = append_the_last_segment_data_and_delete_last_step_keys(data, other_data_keys)

            # Do some assertion checks
            assert data['vehicle_state_nk3'].shape[0] == (data['episode_number_n1'][-1] - data['episode_number_n1'][0] + 1) + data['episode_number_n1'].shape[0]
        
        # Save the file
        filename = os.path.join(new_dir, 'file%i.pkl' % (j + 1))
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Store the metadata
        if store_metadata:
            metadata_key = os.path.join(new_dir, metadata_prefix, 'file%i.pkl' % (j + 1))
            metadata[metadata_key] = data['vehicle_state_nk3'].shape[0]
            
    # Save the metadata file
    if store_metadata:
        filename = os.path.join(new_dir, 'metadata.pkl')
        with open(filename, 'wb') as handle:
            pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def keep_the_successful_episodes_data(data, keys_to_operate_on, indices_to_keep):
    for key in keys_to_operate_on:
        data[key] = data[key][indices_to_keep]
    return data


def append_the_last_segment_data_and_delete_last_step_keys(data, data_keys_to_append):
    for key in data_keys_to_append:
        last_step_key = 'last_step_' + key
        # This if-else conditions are implemented because of a bug in the collection of last step data
        if key in ['goal_position_n2', 'goal_position_ego_n2']:
            try:
                data[key] = np.concatenate((data[key], data[last_step_key][:, 0, 0:2]), axis=0)
            except IndexError:
                data[key] = np.concatenate((data[key], data[last_step_key]), axis=0)
        elif key in ['optimal_waypoint_n3', 'optimal_waypoint_ego_n3']:
            try:
                data[key] = np.concatenate((data[key], data[last_step_key][:, 0]), axis=0)
            except ValueError:
                data[key] = np.concatenate((data[key], data[last_step_key]), axis=0)
        else:
            data[key] = np.concatenate((data[key], data[last_step_key]), axis=0)
        del data[last_step_key]
    return data


if __name__ == '__main__':
    load_and_correct_pickle_files()
