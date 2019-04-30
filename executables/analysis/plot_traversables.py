import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

traversable_dir = '/home/ext_drive/somilb/data/stanford_building_parser_dataset/traversibles'

def plot_traversables(traversable_dir):
    areas = os.listdir(traversable_dir)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    for area in areas:
        traversable_datafile = os.path.join(traversable_dir, area, 'data.pkl')

        with open(traversable_datafile, 'rb') as f:
            data = pickle.load(f)

        ax.clear()
        
        shape = data['traversible'].shape
        dx = data['resolution']/100.
        extent = (0.0, shape[1]*dx, 0.0, shape[0]*dx)
        
        ax.imshow(data['traversible'], cmap='gray', vmin=-.5, vmax=1.5,
                  extent=extent, origin='lower')
        ax.grid(True)

        figname = os.path.join(traversable_dir, area, 'traversable.pdf')
        fig.savefig(figname, bbox_inches='tight', pad_inches=0, dpi=200)
        

if __name__ == '__main__':
    plot_traversables(traversable_dir)
