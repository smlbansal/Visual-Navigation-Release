import os
import pickle
import time

def load_data(dirname):
    start_time = time.time()
    filenames = os.listdir(dirname)
    for filename in filenames:
        if '.pkl' in filename:
            full_filename = os.path.join(dirname, filename)
            with open(full_filename, 'rb') as f:
                data = pickle.load(f)
    end_time = time.time()
    print('Time Elapsed {:.3f}'.format(end_time-start_time))

if __name__ == '__main__':
    dirname = '/home/vtolani/Documents/Projects/visual_mpc/data/control_pipelines/control_pipeline_v0/planning_horizon_120_dt_0.05/dubins_v2/uniform_grid_n_21734_theta_bins_21_bound_min_0.00_-2.50_-1.57_bound_max_2.50_2.50_1.57/61_velocity_bins'
    load_data(dirname)
    dirname = '/home/ext_drive/somilb/data/control_pipelines/control_pipeline_v0/planning_horizon_120_dt_0.05/dubins_v2/uniform_grid_n_21734_theta_bins_21_bound_min_0.00_-2.50_-1.57_bound_max_2.50_2.50_1.57/61_velocity_bins'
    load_data(dirname)
