from obstacles.obstacle_map import ObstacleMap
import tensorflow as tf
from turtlebot.turtlebot_hardware import TurtlebotHardware


class TurtlebotMap(ObstacleMap):

    def __init__(self, params):
        """
        Initialize a map for the Turtlebot
        """
        self.p = params
        self.imager = TurtlebotHardware.get_hardware_interface(params.hardware_params) 

    def dist_to_nearest_obs(self, pos_nk2):
        with tf.name_scope('dist_to_obs'):
            return tf.zeros_like(pos_nk2[:, :, 0])

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Render the robot's current observation
        """
        img_mkd = self.imager.image
        return img_mkd[None]
