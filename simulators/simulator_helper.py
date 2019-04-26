import numpy as np
import tensorflow as tf
from utils.angle_utils import angle_normalize


class SimulatorHelper(object):

    def apply_control_open_loop(self, start_config, control_nk2,
                                T, sim_mode='ideal'):
        """
        Apply control commands in control_nk2 in an open loop
        fashion to the system starting from start_config.
        """
        x0_n1d, _ = self.system_dynamics.parse_trajectory(start_config)
        applied_actions = []
        states = [x0_n1d*1.]
        x_next_n1d = x0_n1d*1.
        for t in range(T):
            u_n1f = control_nk2[:, t:t+1]
            x_next_n1d = self.system_dynamics.simulate(x_next_n1d, u_n1f, mode=sim_mode)

            # Append the applied action to the action list
            if sim_mode == 'ideal':
                applied_actions.append(u_n1f)
            elif sim_mode == 'realistic':
                # TODO: This line is intended for a real hardware setup.
                # If running this code on a real robot the user will need to
                # implement hardware.state_dx such that it reflects the current
                # sensor reading of the robot's applied actions
                applied_actions.append(np.array(self.system_dynamics.hardware.state_dx*1.)[None, None])
            else:
                assert(False)

            states.append(x_next_n1d)

        commanded_actions_nkf = tf.concat([control_nk2[:, :T], u_n1f], axis=1)
        u_nkf = tf.concat(applied_actions, axis=1)
        x_nkd = tf.concat(states, axis=1)
        trajectory = self.system_dynamics.assemble_trajectory(x_nkd,
                                                              u_nkf,
                                                              pad_mode='repeat')
        return trajectory, commanded_actions_nkf

    def apply_control_closed_loop(self, start_config, trajectory_ref,
                                  k_array_nTf1, K_array_nTfd, T,
                                  sim_mode='ideal'):
        """
        Apply LQR feedback control to the system to track trajectory_ref
        Here k_array_nTf1 and K_array_nTfd are tensors of dimension
        (n, self.T-1, f, 1) and (n, self.T-1, f, d) respectively.
        """
        with tf.name_scope('apply_control'):
            x0_n1d, _ = self.system_dynamics.parse_trajectory(start_config)
            assert(len(x0_n1d.shape) == 3)  # [n,1,x_dim]
            angle_dims = self.system_dynamics._angle_dims
            commanded_actions_nkf = []
            applied_actions = []
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
                                         angle_normalize(error_t_n1d[:, :, angle_dims:angle_dims+1].numpy()),
                                         error_t_n1d[:, :, angle_dims+1:]],
                                        axis=2)
                fdback_nf1 = tf.matmul(K_array_nTfd[:, t],
                                       tf.transpose(error_t_n1d, perm=[0, 2, 1]))
                u_n1f = u_ref_n1f + tf.transpose(k_array_nTf1[:, t] + fdback_nf1,
                                                 perm=[0, 2, 1])

                x_next_n1d = self.system_dynamics.simulate(x_next_n1d, u_n1f, mode=sim_mode)

                commanded_actions_nkf.append(u_n1f)
                # Append the applied action to the action list
                if sim_mode == 'ideal':
                    applied_actions.append(u_n1f)
                elif sim_mode == 'realistic':
                    # TODO: This line is intended for a real hardware setup.
                    # If running this code on a real robot the user will need to
                    # implement hardware.state_dx such that it reflects the current
                    # sensor reading of the robot's applied actions
                    applied_actions.append(np.array(self.system_dynamics.hardware.state_dx*1.)[None, None])
                else:
                    assert(False)

                states.append(x_next_n1d)

            commanded_actions_nkf.append(u_n1f)
            commanded_actions_nkf = tf.concat(commanded_actions_nkf, axis=1)
            u_nkf = tf.concat(applied_actions, axis=1)
            x_nkd = tf.concat(states, axis=1)
            trajectory = self.system_dynamics.assemble_trajectory(x_nkd,
                                                                  u_nkf,
                                                                  pad_mode='repeat')
            return trajectory, commanded_actions_nkf

    def _compute_time_idx_for_termination_condition(self, vehicle_trajectory, condition):
        """
        For a given trajectory termination condition (i.e. timeout, collision, etc.)
        computes the earliest time index at which this condition is met. Returns
        infinity if a condition is not met.
        """
        if condition == 'Timeout':
            time_idx = self._compute_time_idx_for_timeout(vehicle_trajectory)
        elif condition == 'Collision':
            time_idx = self._compute_time_idx_for_collision(vehicle_trajectory)
        elif condition == 'Success':
            time_idx = self._compute_time_idx_for_success(vehicle_trajectory)
        else:
            raise NotImplementedError

        return time_idx

    def _compute_time_idx_for_timeout(self, vehicle_trajectory):
        """
        If vehicle_trajectory has exceeded episode_horizon,
        return episode_horizon, else return infinity.
        """
        if vehicle_trajectory.k >= self.params.episode_horizon:
            time_idx = tf.constant(self.params.episode_horizon)
        else:
            time_idx = tf.constant(np.inf)
        return time_idx

    def _compute_time_idx_for_collision(self, vehicle_trajectory):
        """
        Compute and return the earliest time index of collision in vehicle
        trajectory. If there is no collision return infinity.
        """
        pos_1k2 = vehicle_trajectory.position_nk2()
        obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
        collisions = tf.where(tf.less(obstacle_dists_1k, 0.0))
        collision_idxs = collisions[:, 1]
        if tf.size(collision_idxs).numpy() != 0:
            time_idx = collision_idxs[0]
        else:
            time_idx = tf.constant(np.inf)
        return time_idx

    def _compute_time_idx_for_success(self, vehicle_trajectory):
        """
        Compute and return the earliest time index of success (reaching the goal region)
        in vehicle trajectory. If there is no collision return infinity.
        """
        dist_to_goal_1k = self._dist_to_goal(vehicle_trajectory)
        successes = tf.where(tf.less(dist_to_goal_1k,
                                     self.params.goal_cutoff_dist))
        success_idxs = successes[:, 1]
        if tf.size(success_idxs).numpy() != 0:
            time_idx = success_idxs[0]
        else:
            time_idx = tf.constant(np.inf)
        return time_idx
