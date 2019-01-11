from systems.dubins_v2 import DubinsV2
from turtlebot.turtlebot_hardware import TurtlebotHardware
import tensorflow as tf


class TurtlebotDubinsV2(DubinsV2):
    """
    A turtlebot approximated as a DubinsV2
    (see systems/dubins_v2.py)
    """
    name = 'dubins_v2'

    def __init__(self, dt, params):
        super(TurtlebotDubinsV2, self).__init__(dt, params)
        self.hardware = TurtlebotHardware.get_hardware_interface(params.hardware_params)

    def simulate(self, x_nk3, u_nk2, t=None):
        """
        Execute a linear and angular velocity command
        on the actual turtlebot system.
        """
        linear_velocity_111 = self._saturate_linear_velocity(u_nk2[:, :, 0])
        angular_velocity_111 = self._saturate_angular_velocity(u_nk2[:, :, 1])
        ros_command = [linear_velocity_111[0, 0].numpy(),
                       angular_velocity_111[0, 0].numpy()]
        self.hardware.apply_command(ros_command)
        next_state_3 = self.hardware.state * 1.
        return tf.constant(next_state_3[None, None], dtype=tf.float32)
