import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import subprocess

#for v4l2-ctl
default_settings = {'brightness': 0,
                    'contrast': 32,
                    'saturation': 60,
                    'hue': 0,
                    'white_balance_temperature_auto': 1,#use auto white balance
                    'gamma': 100,
                    'gain': 0,
                    'power_line_frequency': 1,
                    'white_balance_temperature': 4600,#not active with auto white balance
                    'sharpness': 2,
                    'backlight_compensation': 1,
                    'exposure_auto': 3,#aperture priority
                    'exposure_absolute': 157,#not active with auto exposure
                    'exposure_auto_priority': 1}#auto exposure
#modified settings
default_settings['contrast'] = 38
default_settings['saturation'] = 72
default_settings['sharpness'] = 3
default_settings['gain'] = 35
default_settings['hue'] = -7

default_settings['white_balance_temperature'] = 5500

#change settings
pairs = ['%s=%d'%(k, default_settings[k]) for k in default_settings.keys()]
config_str = ' -c ' + ' -c '.join(pairs)
command = 'v4l2-ctl -d /dev/video1 %s'%(config_str)
subprocess.call(command, shell=True)
subprocess.call('v4l2-ctl -d /dev/video1 --all', shell=True)

#Capture Video from the ELP Wide Angle Video Camera
#and stream it over ROS
cam = cv2.VideoCapture(1)

def close_node():
    cam.release()
    exit(0)

print('Starting ELP Camera Node')
rospy.on_shutdown(close_node)
rospy.init_node('ELP_Camera')
cam_publisher = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=5)
bridge = CvBridge()

while True:
    imaged, img = cam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if imaged:
        image_message = bridge.cv2_to_imgmsg(img, encoding='rgb8')
        cam_publisher.publish(image_message)
        cv2.waitKey(20)
cv2.destroyAllWindows()
