import os
import numpy as np
import cv2
import rospy
from utils.utils import check_dotmap_equality
from utils.angle_utils import angle_normalize
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from kobuki_msgs.msg import BumperEvent
from kobuki_msgs.msg import WheelDropEvent
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError


class TurtlebotHardware():
    hardware_interface = None

    def __init__(self, params):
        self.params = params
        
        self.state = np.zeros(3)
        self.state_dx = np.zeros(2)
        self.num_collision_steps = 0
        self.hit_obstacle = False
        self.raw_image = None
        self.save_images = False
        self.images_saved = 0
        self.img_dir = None
        self.hit_obstacle = False

            
        rospy.init_node('Visual_MPC_Turtlebot_Agent')

        # Initialize Sensors
        self.odom = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.odom_reset = rospy.Publisher('/mobile_base/commands/reset_odometry',
                                          Empty, queue_size=5)
        self.bumper = rospy.Subscriber('/mobile_base/events/bumper',
                                       BumperEvent, self.bump_callback)
        self.wheel_drop = rospy.Subscriber('/mobile_base/events/wheel_drop',
                                           WheelDropEvent,  self.wheel_drop_callback)
        if params.image_type == 'rgb':  # use orbbec astra camera over ros
            self.imager = rospy.Subscriber('/camera/rgb/image_raw', Image, self.imager_callback)
        elif params.image_type == 'depth':
            self.imager = rospy.Subscriber('/camera/depth/image_raw', Image, self.imager_callback)
        else:
            assert(False)
        self.bridge = CvBridge()

        # Initialize Actuators
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

        # Initialize rospy
        rospy.sleep(1)
        self.reset_odom()
        self.r = rospy.Rate(int(1./params.dt))  # Set the actuator frequency in Hz

    @staticmethod
    def get_hardware_interface(params):
        if TurtlebotHardware.hardware_interface is None:
            TurtlebotHardware.hardware_interface = TurtlebotHardware(params)
        else:
            assert(check_dotmap_equality(params,
                                         TurtlebotHardware.hardware_interface.params))
        return TurtlebotHardware.hardware_interface

    # Sensor Callbacks
    def odom_callback(self, data):
        quaternion = (data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                      data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        angle = euler_from_quaternion(quaternion)[2]
        self.state[0] = data.pose.pose.position.x
        self.state[1] = data.pose.pose.position.y
        self.state[2] = angle_normalize(angle)
        self.state_dx[0] = data.twist.twist.linear.x
        self.state_dx[1] = data.twist.twist.angular.z

    # TODO Varun T.: Dont convert every image- might block the cpu
    def imager_callback(self, data):
        try:
            if self.params.image_type == 'rgb':
                self.raw_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
            elif self.params.image_type == 'depth':
                self.raw_image = self.bridge.imgmsg_to_cv2(data, '16UC1')
            else:
                assert(False)
        except CvBridgeError as e:
            print(e)

        if self.save_images:
            self.write_img(os.path.join(self.img_dir,
                                        'img_{:d}.png'.format(self.images_saved)),
                           self.raw_image)
            self.images_saved += 1

    def wheel_drop_callback(self, data):
        """
        If the top part of turtlebot hits
        an obstacle (i.e. table), then
        bump sensor may not register
        but the wheel drop one will
        """
        self.hit_obstacle = True

    def bump_callback(self, data):
        self.hit_obstacle = True

    @property
    def image(self):
        if self.params.image_type == 'rgb':  # cvt flips this internally
            image = cv2.resize(self.raw_image*1.,
                               (self.params.image_size[1], self.params.image_size[0]),
                               interpolation=cv2.INTER_AREA)
        elif self.params.image_type == 'depth':
            raise NotImplementedError
        else:
            assert(False)
        return image

    def start_saving_images(self, img_dir):
        """
        Start recording images while the turtlebot is moving
        around (useful for making videos).
        """
        self.img_dir = img_dir
        self.images_saved = 0
        self.save_images = True

    def stop_saving_images(self):
        self.save_images = False

    def reset_odom(self):
        self.num_collision_steps = 0
        self.odom_reset.publish(Empty())
        rospy.sleep(1)

    def write_img(self, name, image):
        if self.params.image_type == 'rgb':
            cv2.imwrite(name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(name, image.astype(np.uint8))

    # Actuators

    def apply_command(self, u):
        """
        Apply an action u= [linear velocity, angular velocity]
        """
        cmd = Twist()
        if not self.hit_obstacle:
            cmd.linear.x = u[0]
            cmd.angular.z = u[1]
        else:
            cmd.linear.x = 0.0
            cmd.linear.z = 0.0
            self.num_collision_steps += 1
        self.cmd_vel.publish(cmd)
        self.r.sleep()
