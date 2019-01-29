import pickle


def load_one_pickle_file():
    filename = '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid/area3/' \
               'full_episode_random_v1_100k/file1.pkl'
    # Load the file
    with open(filename, 'rb') as handle:
        data_current = pickle.load(handle)
        
    import ipdb; ipdb.set_trace()
    print('start debugging.')


if __name__ == '__main__':
    load_one_pickle_file()