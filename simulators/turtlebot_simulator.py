import os
import numpy as np
import tensorflow as tf
from trajectory.trajectory import SystemConfig
from obstacles.turtlebot_map import TurtlebotMap
from simulators.simulator import Simulator
from utils import utils


class TurtlebotSimulator(Simulator):
    name = 'Turtlebot_Simulator'

    def __init__(self, params):
        assert(params.obstacle_map_params.obstacle_map is TurtlebotMap)
        super(TurtlebotSimulator, self).__init__(params=params)
        self.video_number = None

    def simulate(self):
        self.system_dynamics.hardware.track_states = True
        super(TurtlebotSimulator, self).simulate()
        self.system_dynamics.hardware.track_states = False

    def start_recording_video(self, video_number):
        """ Start recording video on the turtlebot."""
        tmp_dir = './tmp/turtlebot_videos/{:d}'.format(video_number)
        utils.mkdir_if_missing(tmp_dir)
        self.system_dynamics.hardware.start_saving_images(tmp_dir)

    def stop_recording_video(self, video_number, video_filename):
        """ Stop recording video on the turtlebot."""
        self.system_dynamics.hardware.stop_saving_images()

        tmp_dir = './tmp/turtlebot_videos/{:d}'.format(video_number)

        # Convert Images from the episode into a video in the session dir
        video_command = 'ffmpeg -i {:s}/img_%d.png -pix_fmt yuv420p {:s}'.format(tmp_dir, video_filename)
        os.system(video_command)

        # Delete the temporary directory with images in it
        utils.delete_if_exists(tmp_dir)

    def _init_obj_fn(self):
        """
        A dummy objective function which always returns 0.0
        so the robot works with the simulator structure.
        """
        from dotmap import DotMap
        dummy_obj = DotMap()
        dummy_obj.evaluate_function = lambda trajectories: [0.0]
        return dummy_obj

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Return the robot's observation from configuration config
        or pos_nk3.
        """
        return self.obstacle_map.get_observation(config=config, pos_n3=pos_n3, **kwargs)

    def _reset_start_configuration(self, rng):
        """
        Reset the Turtlebot's odometer.
        """
        p = self.params
        self.system_dynamics.hardware.reset_odom()
        # ideally should be 0, but when using gazebo it will be some small nonzero number
        assert(np.linalg.norm(np.array(self.system_dynamics.hardware.state)*1.) < 1e-3)
        self.start_config = self.system_dynamics.init_egocentric_robot_config(dt=p.dt,
                                                                              n=1)

    def _reset_goal_configuration(self, rng):
        """
        Reset the turtlebot goal position
        """
        p = self.params

        assert(p.reset_params.goal_config.position.reset_type == 'custom')
        x, y = p.reset_params.goal_config.position.goal_pos
        pos_112 = np.array([[[x, y]]], dtype=np.float32)
        self.goal_config = SystemConfig(dt=p.dt, n=1, k=1,
                                        position_nk2=pos_112)
        return False

    def _reset_obstacle_map(self, rng):
        """
        For Turtlebot the obstacle map holds
        no environment information and thus
        does not need to be reset
        """
        return False

    def _update_fmm_map(self):
        """
        For Turtlebot there is no fmm map.
        """
        return False

    def _init_obstacle_map(self, rng):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p)

    def _dist_to_goal(self, trajectory):
        """
        Calculate the l2 distance to the goal.
        """
        dist_to_goal_1k = tf.norm(trajectory.position_nk2() - self.goal_config.position_nk2(),
                                  axis=2)
        return dist_to_goal_1k

    def _compute_time_idx_for_collision(self, vehicle_trajectory):
        if self.system_dynamics.hardware.hit_obstacle:
            time_idx = tf.constant(vehicle_trajectory.k)
        else:
            time_idx = tf.constant(self.params.episode_horizon+1)
        return time_idx

    def _render_obstacle_map(self, ax):
        """
        The obstacle map is not known a priori so
        do nothing.
        """
        return None
