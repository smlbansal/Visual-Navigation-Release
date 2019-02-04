import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import pickle
from utils.utils import mkdir_if_missing, delete_if_exists, subplot2
from systems.turtlebot_dubins_v2 import TurtlebotDubinsV2
from systems.dubins_v2 import DubinsV2
from optCtrl.lqr import LQRSolver
from trajectory.trajectory import Trajectory, SystemConfig
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.angle_utils import angle_normalize


#trajectory_dir = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/sbpd_projected_grid/nn_waypoint/resnet_50_v1/include_last_step/only_successful_episodes/session_2019-01-27_00-08-31/test/checkpoint_9/session_2019-01-28_18-10-07/rgb_resnet50_nn_waypoint_simulator/trajectory_data' 


trajectory_dir = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/uniform_grid/nn_control/resnet_50_v1/data_distortion_v1/session_2019-01-21_18-01-22/test/checkpoint_18/session_2019-01-24_16-27-22/rgb_resnet50_nn_control_simulator/trajectory_data'
linear_acc_max = .02
angular_acc_max = 2.0

class PrecomputedLQRRunner(object):

    def __init__(self, trajectory_dir, output_dir, params):
        self.trajectory_dir = trajectory_dir
        self.output_dir = output_dir

        self.system_dynamics = TurtlebotDubinsV2(dt=params.dt, params=params) 
        #self.system_dynamics = DubinsV2(dt=params.dt, params=params) 

        
    def render(self, occupancy_grid, extent, trajectory_ref, trajectory_gazebo, applied_controls,
               axs, start, goal_num):
        def render_trajectory(ax, trajectory, goal_num, name):
            x_n = trajectory['position_nk2'][0, :, 0]
            y_n = trajectory['position_nk2'][0, :, 1]
            theta_n = trajectory['heading_nk1'][0, :, 0]
            render_angle_freq = int(len(theta_n)/25.)
            ax.imshow(occupancy_grid, cmap='gray_r',
                          extent=extent, origin='lower', vmax=1.5, vmin=-.5)
            ax.quiver(x_n[::render_angle_freq], y_n[::render_angle_freq],
                          np.cos(theta_n[::render_angle_freq]), np.sin(theta_n[::render_angle_freq]))
            ax.plot(x_n, y_n, 'r--')
            ax.set_title('Goal {:d}, {:s}'.format(goal_num, name))

            # 5 meters in x and y direction from the starting position
            ax.set_xlim(start[0]-5., start[0]+5)
            ax.set_ylim(start[1]-5., start[1]+5)

        render_trajectory(axs[0], trajectory_ref, goal_num, 'Reference')
        render_trajectory(axs[1], trajectory_gazebo, goal_num, 'Simulation')

        x_ref_n = trajectory_ref['position_nk2'][0, :, 0]
        y_ref_n = trajectory_ref['position_nk2'][0, :, 1]
        theta_ref_n = trajectory_ref['heading_nk1'][0, :, 0]
        x_sim_n = trajectory_gazebo['position_nk2'][0, :, 0]
        y_sim_n = trajectory_gazebo['position_nk2'][0, :, 1]
        theta_sim_n = trajectory_gazebo['heading_nk1'][0, :, 0]

        v_ref_n = trajectory_ref['speed_nk1'][0, :, 0]
        w_ref_n = trajectory_ref['angular_speed_nk1'][0, :, 0]
        v_commanded_n = trajectory_gazebo['speed_nk1'][0, :, 0]
        w_commanded_n = trajectory_gazebo['angular_speed_nk1'][0, :, 0]
        try:
            v_applied_n = applied_controls[:, 0]
            w_applied_n = applied_controls[:, 1]
        except IndexError:
            v_applied_n = []
            w_applied_n = []

        ax = axs[2]
        ax.plot(x_ref_n, 'r--', label='Ref')
        ax.plot(x_sim_n, 'g--', label='Sim')
        ax.set_title('# {:d}, X Position vs Time'.format(goal_num))
        ax.legend()

        ax = axs[3]
        ax.plot(y_ref_n, 'r--', label='Ref')
        ax.plot(y_sim_n, 'g--', label='Sim')
        ax.set_title('# {:d}, Y Position vs Time'.format(goal_num))
        ax.legend()

        ax = axs[4]
        ax.plot(theta_ref_n, 'r--', label='Ref')
        ax.plot(theta_sim_n, 'g--', label='Sim')
        ax.set_title('# {:d}, Theta vs Time'.format(goal_num))
        ax.legend()

        ax = axs[5] 
        ax.plot(v_ref_n, 'r--', label='Ref')
        ax.plot(v_commanded_n, 'b--', label='Commanded')
        ax.plot(v_applied_n, 'g--', label='Applied')
        ax.set_title('# {:d}, Linear Velocity (m/s)'.format(goal_num))
        ax.legend()

        ax = axs[6] 
        ax.plot(w_ref_n, 'r--', label='Ref Loop')
        ax.plot(w_commanded_n, 'b--', label='Commanded')
        ax.plot(w_applied_n, 'g--', label='Applied')
        ax.set_title('# {:d}, Angular Velocity (m/s)'.format(goal_num))
        ax.legend()


    def apply_control_open_loop(self, start_config, trajectory,
                                T, sim_mode='ideal'):
        
        applied_control = []
        control_nk2 = trajectory.speed_and_angular_speed_nk2()
        x0_n1d, _ = self.system_dynamics.parse_trajectory(start_config)
        actions = []
        states = [x0_n1d*1.]
        x_next_n1d = x0_n1d*1.
        for t in range(T):
            u_n1f = control_nk2[:, t:t+1] 
            
            if t == 0:
                linear_acc_n11 = tf.minimum(tf.abs(u_n1f[:, :, 0:1])-start_config.speed_nk1(),
                                           linear_acc_max)
                angular_acc_n11 = tf.minimum(tf.abs(u_n1f[:, :,
                                                          1:2])-start_config.angular_speed_nk1(),
                                            angular_acc_max)

                v_next_n11 = start_config.speed_nk1() + linear_acc_n11 * tf.sign(u_n1f[:, :, 0:1])
                w_next_n11 = start_config.angular_speed_nk1() + angular_acc_n11 * tf.sign(u_n1f[:, :, 1:2])

            else:
                linear_acc_n11 = tf.minimum(tf.abs(u_n1f[:, :, 0:1] - applied_control[-1][0]),
                                           linear_acc_max)
                angular_acc_n11 = tf.minimum(tf.abs(u_n1f[:, :, 1:2] - applied_control[-1][1]),
                                            angular_acc_max)

                v_next_n11 =  applied_control[-1][0] + linear_acc_n11 * tf.sign(u_n1f[:, :, 0:1] -
                                                                         applied_control[-1][0])
                w_next_n11 =  applied_control[-1][1] + angular_acc_n11 * tf.sign(u_n1f[:, :, 1:2] - applied_control[-1][1])
            #print('Linear Acc: {:.3f} Angular Acc: {:.3f}'.format(linear_acc_n11[0, 0, 0].numpy(),
            #                                                      angular_acc_n11[0, 0,
            #                                                                      0].numpy()))

            u_n1f = tf.concat([v_next_n11, w_next_n11], axis=2)
            x_next_n1d = self.system_dynamics.simulate(x_next_n1d, u_n1f, mode=sim_mode)
            try:
                applied_control.append(self.system_dynamics.hardware.state_dx*1.)
            except:
                applied_control.append(u_n1f[0, 0].numpy())
            actions.append(u_n1f)
            states.append(x_next_n1d)
        u_nkf = tf.concat(actions, axis=1)
        x_nkd = tf.concat(states, axis=1)
        trajectory = self.system_dynamics.assemble_trajectory(x_nkd,
                                                              u_nkf,
                                                              pad_mode='repeat')
        try:
            applied_control.append(self.system_dynamics.hardware.state_dx*1.)
        except:
            applied_control.append(u_n1f[0, 0].numpy())
        return trajectory, applied_control


    def apply_control(self, start_config, trajectory,
                      k_array_nTf1, K_array_nTfd, T,
                      sim_mode='ideal'):
        """
        apply the derived control to the error system to derive a new
        trajectory. Here k_array_nTf1 and K_aaray_nTfd are
        tensors of dimension (n, self.T-1, f, 1) and (n, self.T-1, f, d) respectively.
        """
        #print('Start[{.3f}, {:.3f}, {:.3f}]\n')
        #print('\n')
        applied_control = []
        with tf.name_scope('apply_control'):
            x0_n1d, _ = self.system_dynamics.parse_trajectory(start_config)
            assert(len(x0_n1d.shape) == 3)  # [n,1,x_dim]
            angle_dims = self.system_dynamics._angle_dims
            actions = []
            states = [x0_n1d*1.]
            x_ref_nkd, u_ref_nkf = self.system_dynamics.parse_trajectory(trajectory)
            x_next_n1d = x0_n1d*1.
            for t in range(T):
                x_ref_n1d, u_ref_n1f = x_ref_nkd[:, t:t+1], u_ref_nkf[:, t:t+1]
                error_t_n1d = x_next_n1d - x_ref_n1d
            
                # TODO: Currently calling numpy() here as tfe.DEVICE_PLACEMENT_SILENT
                # is not working to place non-gpu ops (i.e. mod) on the cpu
                # turning tensors into numpy arrays is a hack around this.
                error_t_n1d = tf.concat([error_t_n1d[:, :, :angle_dims],
                                         angle_normalize(error_t_n1d[:, :,
                                                                     angle_dims:angle_dims+1].numpy()),
                                         error_t_n1d[:, :, angle_dims+1:]],
                                        axis=2)
                fdback_nf1 = tf.matmul(K_array_nTfd[:, t],
                                       tf.transpose(error_t_n1d, perm=[0, 2, 1]))
                u_n1f = u_ref_n1f + tf.transpose(k_array_nTf1[:, t] + fdback_nf1,
                                                 perm=[0, 2, 1])
                
                if t == 0:
                    linear_acc_n11 = tf.minimum(tf.abs(u_n1f[:, :, 0:1])-start_config.speed_nk1(),
                                               linear_acc_max)
                    angular_acc_n11 = tf.minimum(tf.abs(u_n1f[:, :,
                                                              1:2])-start_config.angular_speed_nk1(),
                                                angular_acc_max)

                    v_next_n11 = start_config.speed_nk1() + linear_acc_n11 * tf.sign(u_n1f[:, :, 0:1])
                    w_next_n11 = start_config.angular_speed_nk1() + angular_acc_n11 * tf.sign(u_n1f[:, :, 1:2])

                else:
                    linear_acc_n11 = tf.minimum(tf.abs(u_n1f[:, :, 0:1] - applied_control[-1][0]),
                                               linear_acc_max)
                    angular_acc_n11 = tf.minimum(tf.abs(u_n1f[:, :, 1:2] - applied_control[-1][1]),
                                                angular_acc_max)

                    v_next_n11 =  applied_control[-1][0] + linear_acc_n11 * tf.sign(u_n1f[:, :, 0:1] -
                                                                             applied_control[-1][0])
                    w_next_n11 =  applied_control[-1][1] + angular_acc_n11 * tf.sign(u_n1f[:, :, 1:2] - applied_control[-1][1])
                #print('Linear Acc: {:.3f} Angular Acc: {:.3f}'.format(linear_acc_n11[0, 0, 0].numpy(),
                #                                                      angular_acc_n11[0, 0,
                #                                                                      0].numpy()))

                u_n1f = tf.concat([v_next_n11, w_next_n11], axis=2)
                x_next_n1d = self.system_dynamics.simulate(x_next_n1d, u_n1f, mode=sim_mode)
                try:
                    applied_control.append(self.system_dynamics.hardware.state_dx*1.)
                except:
                    applied_control.append(u_n1f[0, 0].numpy())
                actions.append(u_n1f)
                states.append(x_next_n1d)
            u_nkf = tf.concat(actions, axis=1)
            x_nkd = tf.concat(states, axis=1)
            trajectory = self.system_dynamics.assemble_trajectory(x_nkd,
                                                                  u_nkf,
                                                                  pad_mode='repeat')
            try:
                applied_control.append(self.system_dynamics.hardware.state_dx*1.)
            except:
                applied_control.append(u_n1f[0, 0].numpy())
            return trajectory, applied_control

    def run_goals(self, n, mode='waypoint'):
        if os.path.exists(self.output_dir):
            assert False, 'Output directory already exists'

        mkdir_if_missing(self.output_dir)
        
        traj_filenames = os.listdir(trajectory_dir)
        traj_filenames.sort(key=lambda x: int(x.split('traj_')[-1].split('.pkl')[0]))
        traj_filenames = traj_filenames[:n]
        
        fig, axss, _ = subplot2(plt, (2*n, 4), (8, 8), (.4, .4))

        for i, traj_filename in enumerate(traj_filenames):
            with open(os.path.join(self.trajectory_dir, traj_filename), 'rb') as f:
                data = pickle.load(f)

            # Load the open loop trajectory
            perfect_trajectory = Trajectory.init_from_numpy_repr(**data['trajectory_info'])
            start_config = SystemConfig.init_config_from_trajectory_time_index(perfect_trajectory, 0)

            # Reset the Odometer and the start state of the gazebo robot
            self.system_dynamics.reset_start_state(start_config)

            current_config = start_config
           
            plans = []
            # Apply LQR Controllers for Each Trajectory Segment
            if mode == 'waypoint':
                n = data['K_nkfd'].shape[0]
                K_nkfd = tf.constant(data['K_nkfd'], dtype=tf.float32)
                k_nkf1 = tf.constant(data['k_nkf1'], dtype=tf.float32)
                planned_trajectory_n = Trajectory.init_from_numpy_repr(**data['planned_trajectory'])
                applied_controls = []
                lqr_segments = []

                for j in range(n):
                    planned_trajectory_segment = planned_trajectory_n[j]
                    K_segment_1kfd = K_nkfd[j:j+1]
                    k_segment_1kf1 = k_nkf1[j:j+1]
                    k = k_segment_1kf1.shape[1].value
                    #print(self.system_dynamics.state_realistic_113.numpy()[0, 0])
                    #print(current_config.position_and_heading_nk3().numpy()[0, 0])
                    #print('\n')
                    
                    lqr_trajectory_segment, applied_control = self.apply_control(current_config,
                                                                                 planned_trajectory_segment,
                                                                                 k_segment_1kf1,
                                                                                 K_segment_1kfd,
                                                                                 sim_mode='realistic',
                                                                                 T=k-1)
                    plans.append(Trajectory.new_traj_clip_along_time_axis(planned_trajectory_segment,
                                                                          k))
                    lqr_segments.append(Trajectory.copy(lqr_trajectory_segment))
                    applied_controls.append(applied_control)
                    current_config = SystemConfig.init_config_from_trajectory_time_index(lqr_trajectory_segment, -1)
                lqr_trajectory = Trajectory.concat_along_time_axis(lqr_segments)
                applied_controls = np.concatenate(applied_controls, axis=0)
            elif mode == 'control':
                lqr_trajectory, applied_control = self.apply_control_open_loop(current_config,
                                                                                 perfect_trajectory,
                                                                                 sim_mode='realistic',
                                                                                 T=perfect_trajectory.k-1)
                applied_controls = np.array(applied_control)
            else:
                assert False
            
            # Save the gazebo trajectories so we dont have to run the script again to visualize
            # them
            data['gazebo_trajectory'] = lqr_trajectory.to_numpy_repr()
            with open(os.path.join(self.output_dir, traj_filename), 'wb') as f:
                pickle.dump(data, f)

            # Plot trajectories
            occupancy_grid = data['occupancy_grid']
            grid_extent = data['map_bounds_extent']
            goal_num = int(traj_filename.split('traj_')[-1].split('.pkl')[0])

            start = start_config.position_and_heading_nk3()[0, 0].numpy()

            #planned_trajectory = Trajectory.concat_along_time_axis(plans)
            self.render(occupancy_grid, grid_extent,
                        perfect_trajectory.to_numpy_repr(),
                        #planned_trajectory.to_numpy_repr(),
                        lqr_trajectory.to_numpy_repr(),
                        applied_controls,
                        np.array([axss[2*i], axss[2*i+1]]).flatten(),
                        start, goal_num)

        img_filename = os.path.join(self.output_dir, 'goals.pdf')
        fig.savefig(img_filename, bbox_inches='tight')

def main():
    from params.system_dynamics.turtlebot_dubins_v2_params import create_params as create_system_params
    import tensorflow as tf
    tf.enable_eager_execution()
    matplotlib.style.use('ggplot')
    
    output_dir = './tmp/gazebo_tune_pid/lqr'
    output_dir = os.path.join(output_dir, 'joint_motor')
    output_dir = os.path.join(output_dir, 'max_force_100',
                              'gazebo_acceleration_linear_acc_.02_waypo_control')

    system_params = create_system_params()
    system_params.hardware_params.image_type = 'rgb'
    system_params.hardware_params.image_size = (224, 224, 3)
    system_params.hardware_params.dt = system_params.dt

    gz = PrecomputedLQRRunner(trajectory_dir, output_dir, system_params)
    gz.run_goals(n=5, mode='control')

if __name__ == '__main__':
    main()
