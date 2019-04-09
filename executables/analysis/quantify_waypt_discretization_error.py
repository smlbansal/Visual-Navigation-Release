import pickle
import numpy as np
import os
from utils.angle_utils import angle_normalize
import matplotlib.pyplot as plt


#data_dir = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/sbpd_projected_grid/nn_waypoint/resnet_50_v1/include_last_step/only_successful_episodes/training_continued_from_epoch9/session_2019-01-27_23-32-01/test/checkpoint_9/record_waypt_prediction_distance_from_grid/session_2019-04-09_14-29-24/rgb_resnet50_nn_waypoint_simulator'

data_dir = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/sbpd_projected_grid/nn_waypoint/resnet_50_v1/include_last_step/only_successful_episodes/training_continued_from_epoch9/session_2019-01-27_23-32-01/test/checkpoint_9/record_waypt_prediction_distance_from_grid/session_2019-04-09_15-39-39/rgb_resnet50_nn_waypoint_simulator'


def plot_differences(error_n):
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111)
    ax.hist(error_n, bins=200)
    ax.set_title('Histogram of L2 Norm of Difference between\n' +\
                  'predicted Waypt (x, y, theta) and precomputed waypt (x, y, theta)')

    fig.savefig('./tmp/waypoint_analysis/waypt_discretization_error.png', bbox_inches='tight',
                pad_inches=0, dpi=200)


def quantify_waypt_discretization_error(data_dir):
    trajectory_dir = os.path.join(data_dir, 'trajectories')
    nn_waypts = []
    disc_waypts = []
    for filename in os.listdir(trajectory_dir):
        if 'metadata' in filename:
            continue

        trajectory_filename = os.path.join(trajectory_dir, filename)
        with open(trajectory_filename, 'rb') as f:
            data = pickle.load(f)

        nn_waypts.append(data['nn_waypts_local'])
        disc_waypts.append(data['disc_waypts_local'])

    nn_waypts_n3 = np.concatenate(nn_waypts, axis=0)[:, 0, 0, :]
    disc_waypts_n3 = np.concatenate(disc_waypts, axis=0)[:, 0, 0, :]

    error_n3 = np.concatenate([(nn_waypts_n3-disc_waypts_n3)[:, :2],
                               angle_normalize((nn_waypts_n3-disc_waypts_n3)[:, 2:3])],
                              axis=1)
    error_n = np.linalg.norm(error_n3, axis=1)

    idx = np.argmax(error_n)
    print('Worst Prediction')
    print('Predicted Waypt: [{:.3f}, {:.3f}, {:.3f}] '.format(*nn_waypts_n3[idx]) +
          'Disc Waypoint: [{:.3f}, {:.3f}, {:.3f}]'.format(*disc_waypts_n3[idx]))
    print('Total Error: {:.3f} '.format(error_n[idx]) + 
          'Linear Error: {:.3f} '.format(np.linalg.norm(error_n3[idx, :2])) + 
          'Angular Error: {:.3f}'.format(np.abs(error_n3[idx, 2])))

    plot_differences(error_n)


if __name__ == '__main__':
    quantify_waypt_discretization_error(data_dir)
