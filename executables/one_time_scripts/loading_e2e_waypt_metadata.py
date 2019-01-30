import pickle
import os
import numpy as np
    

def analyze_data():
    
    # # Area 1 files
    # e2e_filename = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/sbpd_projected_grid/nn_control/resnet_50_v1/' \
    #                'include_last_step/only_successful_episodes/session_2019-01-27_23-34-22/test/checkpoint_18/' \
    #                'session_2019-01-30_13-55-54/rgb_resnet50_nn_control_simulator/trajectories/metadata.pkl'
    #
    # waypoint_filename = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/sbpd_projected_grid/nn_waypoint/' \
    #                     'resnet_50_v1/include_last_step/only_successful_episodes/training_continued_from_epoch9/' \
    #                     'session_2019-01-27_23-32-01/test/checkpoint_9/session_2019-01-30_13-47-21/' \
    #                     'rgb_resnet50_nn_waypoint_simulator/trajectories/metadata.pkl'

    # Area 6 files
    e2e_filename = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/sbpd_projected_grid/nn_control/resnet_50_v1/' \
                   'include_last_step/only_successful_episodes/session_2019-01-27_23-34-22/test/checkpoint_18/' \
                   'session_2019-01-30_14-04-20/rgb_resnet50_nn_control_simulator/trajectories/metadata.pkl'

    waypoint_filename = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/sbpd_projected_grid/nn_waypoint/' \
                        'resnet_50_v1/include_last_step/only_successful_episodes/training_continued_from_epoch9/' \
                        'session_2019-01-27_23-32-01/test/checkpoint_9/session_2019-01-30_13-57-31/' \
                        'rgb_resnet50_nn_waypoint_simulator/trajectories/metadata.pkl'
    
    
    # Load the file
    with open(e2e_filename, 'rb') as handle:
        e2e_data = pickle.load(handle)
    with open(waypoint_filename, 'rb') as handle:
        waypt_data = pickle.load(handle)

    e2e_data['episode_number'] = np.array(e2e_data['episode_number'])
    e2e_data['episode_type_int'] = np.array(e2e_data['episode_type_int'])
    e2e_successful_episodes = e2e_data['episode_number'][np.where(e2e_data['episode_type_int'] == 2)[0]]
    e2e_fail_episodes = np.array(list(set(e2e_data['episode_number']).difference(e2e_successful_episodes)))
    
    waypt_data['episode_number'] = np.array(waypt_data['episode_number'])
    waypt_data['episode_type_int'] = np.array(waypt_data['episode_type_int'])
    waypt_successful_episodes = waypt_data['episode_number'][np.where(waypt_data['episode_type_int'] == 2)[0]]
    waypt_fail_episodes = np.array(list(set(waypt_data['episode_number']).difference(waypt_successful_episodes)))

    e2e_success_waypt_success = np.array(list(set(e2e_successful_episodes).intersection(waypt_successful_episodes)))
    e2e_success_waypt_fail = np.array(list(set(e2e_successful_episodes).intersection(waypt_fail_episodes)))
    e2e_fail_waypt_success = np.array(list(set(e2e_fail_episodes).intersection(waypt_successful_episodes)))
    e2e_fail_waypt_fail = np.array(list(set(e2e_fail_episodes).intersection(waypt_fail_episodes)))
    
    print('Overall success rate for the waypoint method is', 100 * waypt_successful_episodes.shape[0]/waypt_data['episode_number'].shape[0])
    print('Overall success rate for the E2E method is', 100 * e2e_successful_episodes.shape[0]/e2e_data['episode_number'].shape[0])
    print('\n')
    
    print('E2E Successful and Waypoint Successful', e2e_success_waypt_success)
    print('Percentage of such goals is', 100 * e2e_success_waypt_success.shape[0]/e2e_data['episode_number'].shape[0])
    print('\n \n')
    
    print('E2E Successful and Waypoint Failure', e2e_success_waypt_fail)
    print('Percentage of such goals is', 100 * e2e_success_waypt_fail.shape[0]/e2e_data['episode_number'].shape[0])
    print('\n \n')
    
    print('E2E Failure and Waypoint Successful', e2e_fail_waypt_success)
    print('Percentage of such goals is', 100 * e2e_fail_waypt_success.shape[0]/e2e_data['episode_number'].shape[0])
    print('\n \n')
    
    print('E2E Failure and Waypoint Failure', e2e_fail_waypt_fail)
    print('Percentage of such goals is', 100 * e2e_fail_waypt_fail.shape[0]/e2e_data['episode_number'].shape[0])
    print('\n \n')


if __name__ == '__main__':
    analyze_data()
