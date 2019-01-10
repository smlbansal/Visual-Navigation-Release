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
        import pdb; pdb.set_trace()
        with tf.name_scope('simulate'):
            delta_x_nk3 = tf.stack([self._saturate_linear_velocity(u_nk2[:, :, 0])*tf.cos(x_nk3[:, :, 2]),
                                    self._saturate_linear_velocity(u_nk2[:, :, 0])*tf.sin(x_nk3[:, :, 2]),
                                    self._saturate_angular_velocity(u_nk2[:, :, 1])], axis=2)
            return x_nk3 + self._dt*delta_x_nk3
