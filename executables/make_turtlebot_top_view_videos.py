import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
import time
import pickle
import matplotlib.patches as patches
import cv2

top_view_padding = 1.0 # Pad by .5 meters on all sides


def compute_top_view_bounds(trajectory):
    pos_nk2 = trajectory['position_nk2']
    xs = pos_nk2[0, :, 0]
    ys = pos_nk2[0, :, 1]
    return ((np.min(xs) - top_view_padding, np.max(xs) + top_view_padding),
            (np.min(ys) - top_view_padding, np.max(ys) + top_view_padding))

def compute_aspect_ratio(top_view_bounds):
    x_dims = top_view_bounds[0]
    y_dims = top_view_bounds[1]
    width = x_dims[1] - x_dims[0]
    height = y_dims[1] - y_dims[0]
    return height/width

def convert_fov_grid_to_world(pos_2, heading_1, fov_grid_n2):
    theta = heading_1[0]
    R = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    pos_rotated = R.dot(fov_grid_n2.T)
    return pos_rotated.T + [pos_2]

def plot_fov_grid_at_start_pos(ax, pos_2, heading_1, fov_grid_n2):
    fov_grid_points_world = convert_fov_grid_to_world(pos_2, heading_1, fov_grid_n2)
    poly = patches.Polygon(fov_grid_points_world, color='blue', alpha=.2)
    ax.add_patch(poly)

def create_topview_video(input_file, fov_grid_n2, goal_pos, fig_width=10, quiver_freq=50,
                         camera_freq=30, resample=True, stylized_top_view=None):
    assert(os.path.exists(input_file))
   
    file_number = int(input_file.split('/')[-1].split('traj_')[-1].split('.')[0])
    output_dir = os.path.join('/'.join(input_file.split('/')[:-2]), 'top_view_videos')
    tmp_directory = os.path.join(output_dir, '{:.3f}'.format(time.time())) 
    
    utils.mkdir_if_missing(output_dir)
    utils.mkdir_if_missing(tmp_directory)

    with open(input_file, 'rb') as f:
        data = pickle.load(f)


    vehicle_trajectory = data['vehicle_trajectory']
    
    # Resample the signal so it is at 30hz (camera freq)
    if resample:
        dt = vehicle_trajectory['dt']
        traj_freq = int(1./dt)
        resample_factor = 1.*camera_freq/traj_freq
        xs = vehicle_trajectory['position_nk2'][0,:, 0]
        ys = vehicle_trajectory['position_nk2'][0,:, 1]
        thetas = vehicle_trajectory['heading_nk1'][0,:, 0]
        new_k = int(np.ceil(len(xs)*resample_factor))

        old_pts = np.r_[:vehicle_trajectory['k']]*dt
        new_pts = np.r_[:new_k]*(1./camera_freq)
        xs = np.interp(new_pts, old_pts, xs)
        ys = np.interp(new_pts, old_pts, ys)
        thetas = np.interp(new_pts, old_pts, np.unwrap(thetas))
        vehicle_trajectory['position_nk2'] = np.stack([xs, ys], axis=1)[None]
        vehicle_trajectory['heading_nk1'] = thetas[None, :, None] 
        vehicle_trajectory['k'] = new_k

    # Compute the shape of the top view map
    #SDH7
    top_view_bounds = [[-.5, 7.0], [-3.0, 3.0]]
    #top_view_bounds = compute_top_view_bounds(vehicle_trajectory)
    aspect_ratio = compute_aspect_ratio(top_view_bounds)

    # Construct the figure/ axes
    fig = plt.figure(frameon=False)
    fig.set_size_inches(fig_width*1.557, fig_width*aspect_ratio*1.572)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    if stylized_top_view is not None:
        stylized_top_view = cv2.imread(stylized_top_view)[:, :, ::-1]
    #fig = plt.figure(figsize=(fig_width, fig_width*aspect_ratio))
    #ax = fig.add_subplot(111)
    
    for t in range(1, vehicle_trajectory['k']):
        ax.clear()

        if stylized_top_view is not None:
            ax.imshow(stylized_top_view, extent= [top_view_bounds[0][0], top_view_bounds[0][1],
                                                  top_view_bounds[1][0], top_view_bounds[1][1]])
            #import pdb; pdb.set_trace()
        pos_t2 = vehicle_trajectory['position_nk2'][0, :t]
        heading_t1 = vehicle_trajectory['heading_nk1'][0, :t]
        ax.plot(pos_t2[:, 0], pos_t2[:, 1], 'r-', linewidth=6)
        #ax.quiver(pos_t2[:, 0][::quiver_freq], pos_t2[:, 1][::quiver_freq],
        #          np.cos(heading_t1[::quiver_freq]), np.sin(heading_t1[::quiver_freq]),
        #         linewidths=1, width=.1)
        
        # Add the Stylized FOV
        plot_fov_grid_at_start_pos(ax, pos_t2[-1], heading_t1[-1], fov_grid_n2)
        
        # Plot the Start Position and Current Robot State
        ax.plot(pos_t2[0, 0], pos_t2[0, 1], 'b.', markersize=45)
        ax.quiver(pos_t2[-1:, 0], pos_t2[-1, 1], np.cos(heading_t1[-1]),
                  np.sin(heading_t1[-1]))


        # Plot The Goal Region
        ax.plot(goal_pos[0], goal_pos[1], 'k*', markersize=10)
        c = plt.Circle(goal_pos, .3, color='green')
        ax.add_artist(c)

        # Make the axis a predetermined size
        ax.set_xlim(top_view_bounds[0])
        ax.set_ylim(top_view_bounds[1])
        #ax.axis(False)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(os.path.join(tmp_directory, 'img_{:d}.png'.format(t)), bbox_inches='tight',
                   pad_inches=0)

    # Convert Images from the episode into A Video in the session dir
    video_filename = os.path.join(output_dir, 'traj_{:d}.mp4'.format(file_number))
    video_command = 'ffmpeg -i {:s}/img_%d.png -pix_fmt yuv420p {:s}'.format(tmp_directory, video_filename)
    os.system(video_command)  
   
    # Delete the temporary directory
    utils.delete_if_exists(tmp_directory)

def generate_stylized_fov_grid(half_fovh=np.pi/4., max_depth=4.):
    thetas = np.linspace(-half_fovh, half_fovh, 100)
    xs, ys = np.cos(thetas)*max_depth, np.sin(thetas)*max_depth
    pts = np.stack([xs, ys], axis=1)
    return np.concatenate([[[0.0, 0.0]], pts], axis=0)

def main():
    import matplotlib
    matplotlib.style.use('ggplot')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_file', type=str, required=True)
    parser.add_argument('--stylized_top_view', type=str, required=False)
    args = parser.parse_args()
    fov_grid_n2 = generate_stylized_fov_grid()
    create_topview_video(args.input_file, fov_grid_n2, goal_pos=[6.0, 0.0],
                         stylized_top_view=args.stylized_top_view)

if __name__ == '__main__':
    main()
