import pickle
import os

data_dir = '/home/ext_drive/somilb/data/topview_full_episode'

filenames = os.listdir(data_dir)
filenames.sort()

n = 0

for filename in filenames:
    full_filename = os.path.join(data_dir, filename)
    with open(full_filename, 'rb') as f:
        data = pickle.load(f)
        n += data['goal_position_n2'].shape[0]
print(n)
