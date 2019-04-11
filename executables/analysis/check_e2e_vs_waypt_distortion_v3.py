import pickle
import numpy as np
import os

e2e_dir = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/sbpd_projected_grid/nn_control/resnet_50_v1/include_last_step/only_successful_episodes/data_distortion_v3/session_2019-02-07_17-14-50/test/checkpoint_20/cleaned_code_post_rss/session_2019-02-12_10-26-00/rgb_resnet50_nn_control_simulator'

waypt_dir = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/sbpd_projected_grid/nn_waypoint/resnet_50_v1/include_last_step/only_successful_episodes/data_distortion_v3/session_2019-01-28_21-02-53/test/checkpoint_20/cleaned_code_post_rss/session_2019-02-12_10-59-36/rgb_resnet50_nn_waypoint_simulator'

def compare_e2e_vs_waypt(e2e_dir, waypt_dir):
    e2e_metadata_file = os.path.join(e2e_dir, 'trajectories', 'metadata.pkl')
    with open(e2e_metadata_file, 'rb') as f:
        e2e_data = pickle.load(f)
    successful_idxs = np.where(np.array(e2e_data['episode_type_string']) == 'Success')[0]
    successful_e2e_episode_numbers = np.array(e2e_data['episode_number'])[successful_idxs]

    unsuccessful_idxs = np.where(np.array(e2e_data['episode_type_string']) != 'Success')[0]
    unsuccessful_e2e_episode_numbers = np.array(e2e_data['episode_number'])[unsuccessful_idxs]

    waypt_metadata_file = os.path.join(waypt_dir, 'trajectories', 'metadata.pkl')
    with open(waypt_metadata_file, 'rb') as f:
        waypt_data = pickle.load(f)
    successful_idxs = np.where(np.array(waypt_data['episode_type_string']) == 'Success')[0]
    successful_waypt_episode_numbers = np.array(waypt_data['episode_number'])[successful_idxs]

    unsuccessful_idxs = np.where(np.array(waypt_data['episode_type_string']) != 'Success')[0]
    unsuccessful_waypt_episode_numbers = np.array(waypt_data['episode_number'])[unsuccessful_idxs]

    # Compute the goals where they both succeed or both fail
    common_successes = np.intersect1d(successful_e2e_episode_numbers,
                                      successful_waypt_episode_numbers)

    common_failures = np.intersect1d(unsuccessful_e2e_episode_numbers,
                                     unsuccessful_waypt_episode_numbers)
    print('Common Successes: {:s}'.format(np.array2string(common_successes)))
    print('Common Failures: {:s}'.format(np.array2string(common_failures)))
    print('{:d} Total Goals'.format(len(data['episode_number'])))


    # Compute the goals where they disagree
    all_goals_set = set(e2e_data['episode_number'])
    common_successes = set(common_successes)
    common_failures = set(common_failures)

    [all_goals_set.remove(x) for x in common_successes]
    [all_goals_set.remove(x) for x in common_failures]
    import pdb; pdb.set_trace()
    pass

if __name__ == '__main__':
    compare_e2e_vs_waypt(e2e_dir, waypt_dir)
