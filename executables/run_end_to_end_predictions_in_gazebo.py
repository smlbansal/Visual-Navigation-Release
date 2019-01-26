import matplotlib
matplotlib.use('Agg')
import rospy
import numpy as np
import os
import pickle
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from utils.angle_utils import angle_normalize
from utils.utils import mkdir_if_missing, delete_if_exists, subplot2
from systems.dubins_car import DubinsCar
import matplotlib.pyplot as plt


#trajectory_dir = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/uniform_grid/nn_control/resnet_50_v1/data_distortion_v1/session_2019-01-21_18-01-22/test/checkpoint_18/session_2019-01-24_13-58-44/rgb_resnet50_nn_control_simulator/trajectory_data'
trajectory_dir = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/uniform_grid/nn_control/resnet_50_v1/data_distortion_v1/session_2019-01-21_18-01-22/test/checkpoint_18/session_2019-01-24_16-27-22/rgb_resnet50_nn_control_simulator/trajectory_data'
#trajectory_dir = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/uniform_grid/nn_control/resnet_50_v1/data_distortion_v1/session_2019-01-21_18-01-22/test/checkpoint_18/session_2019-01-24_15-41-26/rgb_resnet50_nn_control_simulator/trajectory_data'

output_dir = './tmp/gazebo_tests/velocity_joint_motors_driver_pid_p_.02_i_2e-5_d_4e-3'


class GazeboTrajectoryRunner(object):

    def __init__(self, trajectory_dir, output_dir, dt=.05):
        self.trajectory_dir = trajectory_dir
        self.output_dir = output_dir

        self.state = np.zeros(3)
        self.state_dx = np.zeros(2)
        
        rospy.init_node('OpenLoopTurtlebotTester')
        self.odom = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.odom_reset = rospy.Publisher('/mobile_base/commands/reset_odometry',
                                          Empty, queue_size=5)

        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
        
        rospy.sleep(1)
        self.reset_odom()
        self.r = rospy.Rate(int(1./dt))

    def odom_callback(self, data):
        quaternion = (data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                      data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        angle = euler_from_quaternion(quaternion)[2]
        self.state[0] = data.pose.pose.position.x
        self.state[1] = data.pose.pose.position.y
        self.state[2] = angle_normalize(angle)
        self.state_dx[0] = data.twist.twist.linear.x
        self.state_dx[1] = data.twist.twist.angular.z

    def reset_odom(self):
        self.odom_reset.publish(Empty())
        rospy.sleep(1)

    def apply_command(self, u):
        """
        Apply an action u= [linear velocity, angular velocity]
        """
        cmd = Twist()
        cmd.linear.x = u[0]
        cmd.angular.z = u[1]
        self.cmd_vel.publish(cmd)
        self.r.sleep()

    def render(self, occupancy_grid, extent, trajectory, axs, start, goal_num, name=''):
        [ax.clear() for ax in axs]

        x_n = trajectory['position_nk2'][0, :, 0]
        y_n = trajectory['position_nk2'][0, :, 1]
        theta_n = trajectory['heading_nk1'][0, :, 0]
        v_n = trajectory['speed_nk1'][0, :, 0]
        w_n = trajectory['angular_speed_nk1'][0, :, 0]
        render_angle_freq = int(len(theta_n)/25.)

        axs[0].imshow(occupancy_grid, cmap='gray_r',
                      extent=extent, origin='lower', vmax=1.5, vmin=-.5)
        axs[0].quiver(x_n[::render_angle_freq], y_n[::render_angle_freq],
                      np.cos(theta_n[::render_angle_freq]), np.sin(theta_n[::render_angle_freq]))
        axs[0].plot(x_n, y_n, 'r--')
        axs[0].set_title('Goal {:d}, {:s}'.format(goal_num, name))
        
        # 5 meters in x and y direction from the starting position
        axs[0].set_xlim(start[0]-5., start[0]+5)
        axs[0].set_ylim(start[1]-5., start[1]+5)

        axs[1].plot(v_n, 'r--')
        axs[1].set_title('# {:d}, Linear Velocity (m/s)'.format(goal_num))

        axs[2].plot(w_n, 'r--')
        axs[2].set_title('# {:d}, Angular Velocity (m/s)'.format(goal_num))

    def run_goals(self, n):
        if os.path.exists(self.output_dir):
            assert False, 'Output directory already exists'

        mkdir_if_missing(self.output_dir)
        
        traj_filenames = os.listdir(trajectory_dir)
        traj_filenames.sort(key=lambda x: int(x.split('traj_')[-1].split('.pkl')[0]))
        traj_filenames = traj_filenames[:n]
        
        fig, axss, _ = subplot2(plt, (2, 3), (8, 8), (.4, .4))

        for i, traj_filename in enumerate(traj_filenames):
            with open(os.path.join(self.trajectory_dir, traj_filename), 'rb') as f:
                data = pickle.load(f)

            # Load the open loop trajectory
            trajectory = data['trajectory_info']
            states_1k3 = np.concatenate([trajectory['position_nk2'], trajectory['heading_nk1']], axis=2)
            controls_1k2 = np.concatenate([trajectory['speed_nk1'], trajectory['angular_speed_nk1']], axis=2)

            # Reset the Odometer
            self.reset_odom()

            # Apply the controls and measure the resulting states
            states = [self.state*1.]
            states_dx = [self.state_dx*1.]
            for t in range(controls_1k2.shape[1]):
                u = controls_1k2[0, t]
                self.apply_command(u)
                states.append(self.state*1.)
                states_dx.append(self.state_dx*1.)

            states = np.array(states)
            states_dx = np.array(states_dx)
            
            # Convert this trajectory to world coordinates
            init_state_113 = states_1k3[:, 0:1]
            states_gazebo_ego_1k3 = states[None].astype(np.float32)
            states_gazebo_world_1k3 = DubinsCar.convert_position_and_heading_to_world_coordinates(init_state_113,
                                                                                                  states_gazebo_ego_1k3).numpy()
            gazebo_traj = {'position_nk2': states_gazebo_world_1k3[:, :, :2],
                           'heading_nk1': states_gazebo_world_1k3[:, :, 2:3],
                           'speed_nk1': states_dx[None, :, 0:1],
                           'angular_speed_nk1': states_dx[None, :, 1:2]}

            # Save the gazebo trajectories so we dont have to run the script again to visualize
            # them
            data['gazebo_trajectory'] = gazebo_traj
            with open(os.path.join(self.output_dir, traj_filename), 'wb') as f:
                pickle.dump(data, f)

            # Plot trajectories
            occupancy_grid = data['occupancy_grid']
            grid_extent = data['map_bounds_extent']
            goal_num = int(traj_filename.split('traj_')[-1].split('.pkl')[0])

            start = init_state_113[0, 0]

            self.render(occupancy_grid, grid_extent, trajectory, axss[0], start, goal_num,
                        name='simulation')

            self.render(occupancy_grid, grid_extent, gazebo_traj, axss[1], start,
                        goal_num, name='gazebo')

            img_filename = os.path.join(self.output_dir, 'goal_{:d}.pdf'.format(goal_num))
            fig.savefig(img_filename, bbox_inches='tight')

def main():
    import tensorflow as tf
    tf.enable_eager_execution()
    matplotlib.style.use('ggplot')
    gz = GazeboTrajectoryRunner(trajectory_dir, output_dir, .05)
    gz.run_goals(n=10)

if __name__ == '__main__':
    main()
