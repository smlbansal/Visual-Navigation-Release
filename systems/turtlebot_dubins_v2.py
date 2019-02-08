from systems.dubins_v2 import DubinsV2
from turtlebot.turtlebot_hardware import TurtlebotHardware
import tensorflow as tf
import numpy as np


class TurtlebotDubinsV2(DubinsV2):
    """
    A turtlebot approximated as a DubinsV2
    (see systems/dubins_v2.py)
    """
    name = 'dubins_v2'

    def __init__(self, dt, params):
        super(TurtlebotDubinsV2, self).__init__(dt, params)
        self.origin_world_113 = np.zeros((1, 1, 3), dtype=np.float32)
        self.hardware = TurtlebotHardware.get_hardware_interface(params.hardware_params)

    def reset_start_state(self, start_config):
        """
        Reset the turtlebot hardware state to 0's. Update the turtlebot
        hardware driver to transform coordinates to the system where start_config
        is at the origin.
        ."""
        self.hardware.reset_odom()
        self.origin_world_113 = start_config.position_and_heading_nk3().numpy()

    @property
    def state_realistic_113(self):
        next_state_113 = self.hardware.state[None, None]*1.

        # Convert the Hardware Sensor Readings to the World Coordinate System
        next_state_113 = self.convert_position_and_heading_to_world_coordinates(self.origin_world_113,
                                                                                next_state_113)
        return next_state_113

    def _simulate_realistic(self, x_nk3, u_nk2, t=None):
        """
        Execute a linear and angular velocity command
        on the actual turtlebot system.
        """
        linear_velocity_111 = self._saturate_linear_velocity(u_nk2[:, :, 0])
        angular_velocity_111 = self._saturate_angular_velocity(u_nk2[:, :, 1])
        ros_command = [linear_velocity_111[0, 0].numpy(),
                       angular_velocity_111[0, 0].numpy()]
        self.hardware.apply_command(ros_command)
        next_state_113 = self.state_realistic_113
        return tf.constant(next_state_113, dtype=tf.float32)
