from utils import utils
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

data_dir = '/home/ext_drive/somilb/data/topview_full_episode'


def main():
    collected_data = []
    filenames = os.listdir(data_dir)
    filenames.sort()
    for filename in filenames:
        full_filename = os.path.join(data_dir, filename)
        with open(full_filename, 'rb') as f:
            data = pickle.load(f)
        collected_data.append(data['vehicle_controls_nk2'])    
    collected_data_nk2 = np.concatenate(collected_data, axis=0)
    collected_data_n2 = collected_data_nk2.reshape(-1, 2)

    collected_v = collected_data_n2[:, 0]
    collected_w = collected_data_n2[:, 1]

    fig, _, axs = utils.subplot2(plt, (2, 2), (8, 8), (.4, .4))
    axs = axs[::-1]

    ax = axs[0]
    ax.hist(collected_v, bins=61, range=(0.0, .6), density=True)
    ax.set_title('Collected Velocity Histogram')

    ax = axs[1]
    ax.hist(collected_w, bins=241, range=(-1.2, 1.2), density=True)
    ax.set_title('Collected Omega Histogram')


    expert_dir = './tmp/expert_data_distribution'
    filename = os.path.join(expert_dir, 'data.pkl')
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    ax = axs[2]
    ax.hist(data['v'][0, :, 0], bins=61, range=(0.0, .6), density=True)
    ax.set_title('Expert Velocity Histogram')

    ax = axs[3]
    ax.hist(data['w'][0, :, 0], bins=241, range=(-1.2, 1.2), density=True)
    ax.set_title('Expert Omega Histogram')

    figname = os.path.join(expert_dir, 'velocity_profiles.png')
    fig.savefig(figname, bbox_inches='tight')



if __name__ == '__main__':
    plt.style.use('ggplot')
    main()
