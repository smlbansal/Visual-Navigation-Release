import numpy as np
import tensorflow as tf
from trajectory.trajectory import Trajectory
from utils.angle_utils import angle_normalize


class SimulatorHelper(object):

    def apply_control_open_loop(self, start_config, control_nk2,
                                T, sim_mode='ideal'):
        x0_n1d, _ = self.system_dynamics.parse_trajectory(start_config)
        actions = []
        states = [x0_n1d*1.]
        x_next_n1d = x0_n1d*1.
        for t in range(T):
            u_n1f = control_nk2[:, t:t+1]
            x_next_n1d = self.system_dynamics.simulate(x_next_n1d, u_n1f, mode=sim_mode)

            # Append the applied action to the action list
            if sim_mode == 'ideal':
                actions.append(u_n1f)
            elif sim_mode == 'realistic':
                actions.append(np.array(self.system_dynamics.hardware.state_dx*1.)*[None, None])
            else:
                assert(False)

            states.append(x_next_n1d)

        u_nkf = tf.concat(actions, axis=1)
        x_nkd = tf.concat(states, axis=1)
        trajectory = self.system_dynamics.assemble_trajectory(x_nkd,
                                                              u_nkf,
                                                              pad_mode='repeat')
        return trajectory

    def apply_control_closed_loop(self, start_config, trajectory_ref,
                                  k_array_nTf1, K_array_nTfd, T,
                                  sim_mode='ideal'):
        """
        Apply LQR feedback control to the system to track trajectory_ref
        Here k_array_nTf1 and K_aaray_nTfd are tensors of dimension
        (n, self.T-1, f, 1) and (n, self.T-1, f, d) respectively.
        """

        p = self.params.system_dynamics_params
        with tf.name_scope('apply_control'):
            x0_n1d, _ = self.system_dynamics.parse_trajectory(start_config)
            assert(len(x0_n1d.shape) == 3)  # [n,1,x_dim]
            angle_dims = self.system_dynamics._angle_dims
            actions = []
            states = [x0_n1d*1.]
            x_ref_nkd, u_ref_nkf = self.system_dynamics.parse_trajectory(trajectory_ref)
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
               
                x_next_n1d = self.system_dynamics.simulate(x_next_n1d, u_n1f, mode=sim_mode)
                
                # Append the applied action to the action list
                if sim_mode == 'ideal':
                    actions.append(u_n1f)
                elif sim_mode == 'realistic':
                    actions.append(np.array(self.system_dynamics.hardware.state_dx*1.)*[None, None])
                else:
                    assert(False)

                states.append(x_next_n1d)

            u_nkf = tf.concat(actions, axis=1)
            x_nkd = tf.concat(states, axis=1)
            trajectory = self.system_dynamics.assemble_trajectory(x_nkd,
                                                                  u_nkf,
                                                                  pad_mode='repeat')
            return trajectory

    def _clip_along_time_axis(self, traj, data, horizon, mode='new'):
        """ Clip a trajectory and the associated planner data
        along the time axis to length horizon."""

        self.planner.clip_data_along_time_axis(data, horizon)

        # Avoid duplicating new trajectory objects
        # as this is unnecesarily slow
        if 'trajectory' in data:
            traj = data['trajectory']
        else:
            if mode == 'new':
                traj = Trajectory.new_traj_clip_along_time_axis(traj, horizon)
            elif mode == 'update':
                traj.clip_along_time_axis(horizon)
            else:
                assert(False)

        return traj, data

    def _compute_time_idx_for_termination_condition(self, vehicle_trajectory, condition):
        """ For a given trajectory termination condition (i.e. timeout, collision, etc.)
        computes the earliest time index at which this condition is met. Returns
        episode_horizon+1 otherwise."""
        if condition == 'Timeout':
            time_idx = tf.constant(self.params.episode_horizon)
        elif condition == 'Collision':
            time_idx = self._compute_time_idx_for_collision(vehicle_trajectory)
        elif condition == 'Success':
            time_idx = self._compute_time_idx_for_success(vehicle_trajectory)
        else:
            raise NotImplementedError

        return time_idx

    def _compute_time_idx_for_collision(self, vehicle_trajectory):
        time_idx = tf.constant(self.params.episode_horizon+1)
        pos_1k2 = vehicle_trajectory.position_nk2()
        obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
        collisions = tf.where(tf.less(obstacle_dists_1k, 0.0))
        collision_idxs = collisions[:, 1]
        if tf.size(collision_idxs).numpy() != 0:
            time_idx = collision_idxs[0]
        return time_idx

    def _compute_time_idx_for_success(self, vehicle_trajectory):
        time_idx = tf.constant(self.params.episode_horizon+1)
        dist_to_goal_1k = self._dist_to_goal(vehicle_trajectory)
        successes = tf.where(tf.less(dist_to_goal_1k,
                                     self.params.goal_cutoff_dist))
        success_idxs = successes[:, 1]
        if tf.size(success_idxs).numpy() != 0:
            time_idx = success_idxs[0]
        return time_idx
