import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

session_dir = '/home/vtolani/Documents/Projects/visual_mpc/tmp/waypoint_analysis/session_2019-04-02_16-14-20/rgb_resnet50_nn_waypoint_simulator'

def analyze_waypoints(trajectory_data):
    waypt_in_obs_indicator_n = waypoints_in_obs(trajectory_data['vehicle_data']['waypoint_config'],
                                                trajectory_data['occupancy_grid'])
    occupied_counter = np.sum(waypt_in_obs_indicator_n)
    return occupied_counter, waypt_in_obs_indicator_n.shape[0]

def waypoints_in_obs(waypoint_configs, occupancy_grid):
    waypt_pos_nk2 = waypoint_configs['position_nk2']

    waypt_pos_grid_space_n2 = (waypt_pos_nk2[:, 0, :]/.05).astype(np.int32)
    in_obs = []
    for waypt_2 in waypt_pos_grid_space_n2:
        if occupancy_grid[waypt_2[1], waypt_2[0]] == 1:
            in_obs.append(True)
        else:
            in_obs.append(False)
    return np.array(in_obs).astype(np.float32)


def plot_for_debugging(trajectory_data, episode_number):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    ax.imshow(trajectory_data['occupancy_grid'], extent=trajectory_data['map_bounds_extent'],
              origin='lower', cmap='gray_r', vmin=-.5, vmax=1.5)

    waypt_in_obs_indicator_n = waypoints_in_obs(trajectory_data['vehicle_data']['waypoint_config'],
                                                trajectory_data['occupancy_grid'])

    waypt_k2 = trajectory_data['vehicle_data']['waypoint_config']['position_nk2'][:, 0, :]

    waypt_obs_free_k2 = waypt_k2[np.logical_not(waypt_in_obs_indicator_n)]
    waypt_in_obs_k2= waypt_k2[waypt_in_obs_indicator_n.astype(np.bool)]
    ax.plot(waypt_obs_free_k2[:, 0], waypt_obs_free_k2[:, 1], 'm^', label='free space')
    ax.plot(waypt_in_obs_k2[:, 0], waypt_in_obs_k2[:, 1], 'g^', label='collision')

    img_dir = '/home/vtolani/Documents/Projects/visual_mpc/tmp/waypoint_analysis/plots_for_debugging'

    # Set xlim and ylim to be +- x meters
    # of the robot starting position
    x = 8.
    start_2 = trajectory_data['vehicle_trajectory']['position_nk2'][0, 0]
    ax.set_xlim(start_2[0]-x, start_2[0]+x)
    ax.set_ylim(start_2[1]-x, start_2[1]+x)
    ax.legend()

    fig.savefig(os.path.join(img_dir, 'img_{:d}.png'.format(episode_number)), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def analyze_waypoints_in_obstacle(session_dir, plot=True): 
    trajectory_filenames = os.listdir(os.path.join(session_dir, 'trajectories'))
    in_obs = {}
    total_waypt = {}
    for filename in trajectory_filenames:
        trajectory_filename = os.path.join(session_dir, 'trajectories', filename)
        if 'metadata' in filename:
            with open(trajectory_filename, 'rb') as f:
                metadata = pickle.load(f)
                continue

        with open(trajectory_filename, 'rb') as f:
            trajectory_data = pickle.load(f)
            num_waypoints_in_obstacle, total_num_waypoints = analyze_waypoints(trajectory_data)
            episode_number = int(filename.split('.pkl')[0].split('traj_')[1])
            in_obs[episode_number] = num_waypoints_in_obstacle
            total_waypt[episode_number] = total_num_waypoints
   
            if plot:
                # Plot Waypts for debugging
                plot_for_debugging(trajectory_data, episode_number)
    

    # Compute Percent of Waypoints in Obstacle
    # for all goals and successful goals
    percent_in_obs_total = np.sum(list(in_obs.values()))/np.sum(list(total_waypt.values()))

    # Filter by Successful Episodes
    successful_episode_numbers = np.array(metadata['episode_number'])[np.array(metadata['episode_type_string']) == 'Success']
    in_obs_successful = {key: in_obs[key] for key in successful_episode_numbers}
    total_waypt_successful = {key: total_waypt[key] for key in successful_episode_numbers}
    
    percent_in_obs_successful = np.sum(list(in_obs_successful.values()))/np.sum(list(total_waypt_successful.values()))
    print('Total Waypoints for, All Goals: {:d}, Successful Goals: {:d}'.format(np.sum(list(total_waypt.values())),
                                                                                np.sum(list(total_waypt_successful.values()))))
    print('Percent Waypoints in Obstacle for All Goals: {:.3f}%'.format(percent_in_obs_total*100.))
    print('Percent Waypoints in Obstacle for Successful Goals: {:.3f}%'.format(percent_in_obs_successful*100.))

    # Draw the Bar Chart
    num_pts_in_obs = list(set(list(in_obs.values())))
    goals_for_num_pts_in_obs = {key:[] for key in num_pts_in_obs}
    [goals_for_num_pts_in_obs[in_obs[key]].append(key) for key in in_obs.keys()]

    percent_success_per_waypt_in_obs = {}
    for key in goals_for_num_pts_in_obs:
        success = [x in successful_episode_numbers for x in goals_for_num_pts_in_obs[key]]
        percent_success_per_waypt_in_obs[key] = np.mean(success)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    y_pos = np.r_[:len(percent_success_per_waypt_in_obs.keys())]
    rects = ax.bar(y_pos, percent_success_per_waypt_in_obs.values(), align='center', alpha=.5)
    
    # Label the rectangles
    for percent, rect, num_goals in zip(percent_success_per_waypt_in_obs.values(), rects,
                                        [len(x) for x in goals_for_num_pts_in_obs.values()]):
        height = rect.get_height()
        text = '{:.2f}%\n({:d} Goals)'.format(percent*100., num_goals)
        ax.text(rect.get_x() + rect.get_width()*.5, 1.01*height,
                text, ha='center', va='bottom')
    
    ax.set_xticks(y_pos)
    ax.set_xlabel('# Of Waypoints in Obstacle over Trajectory')
    ax.set_ylabel('Percent Success')
    img_dir = '/home/vtolani/Documents/Projects/visual_mpc/tmp/waypoint_analysis/' 
    fig.savefig(os.path.join(img_dir, 'barchart.png'), bbox_inches='tight')

if __name__ == '__main__':
    analyze_waypoints_in_obstacle(session_dir, plot=False)
