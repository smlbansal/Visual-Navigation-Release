Visual MPC
==========
### Setup A Virtual Environment
```
conda create -n venv-mpc python=3.6
source activate venv-mpc
pip install -U pip
pip install -r requirements.txt
pip install --upgrade tensorflow (make sure to install v1.10.1)
```


### Setup the Stanford Building Parser Dataset Renderer
```
1. Clone the repository https://github.com/vtolani95/mp-env
2. Checkout branch visual_mpc and follow the setup instructions
3. Install mp-env as a package ("pip install -e ." from the root directory)
```


### Run a simulation-trained policy on the Turtlebot

#### Setup the Turtlebot
```
0. Create a new anaconda environment with python 2.7 (rospy is only built for py27). Follow the instructions above for 'Setup a Virtual Environment'
1. Install ros-kinetic
2. Install the turtlebot drivers for ros-kinetic
    - sudo apt-get install ros-kinetic-turtlebot ros-kinetic-turtlebot-apps ros-kinetic-turtlebot-interactions ros-kinetic-turtlebot-simulator ros-kinetic-kobuki-ftdi ros-kinetic-ar-track-alvar-msgs
4. Install rospkg, defusedxml, numpy with pip
3. Install camera drivers if needed (i.e. for Orbecc Astra)
```

#### Run a Policy on the Turtlebot
```
0. Launch the turtlebot base and camera drivers with
    - roslaunch turtlebot_bringup minimal.launch
    - roslaunch turtlebot_bringup 3dsensor.launch
1. Launch a preatrained waypoint-based navigator on the real robot (uses the resnet50 architecture) 
- PYTHONPATH='/opt/ros/kinetic/lib/python2.7/dist-packages:.' python executables/rgb/resnet50/rgb_waypoint_runner.py test --job-dir ./logs/tmp --params params/turtlebot/turtlebot_waypoint_navigator_params.py -d 0

2. Launch a preatrained end_to_end-based navigator on the real robot (uses the resnet50 architecture) 
- PYTHONPATH='/opt/ros/kinetic/lib/python2.7/dist-packages:.' python executables/rgb/resnet50/rgb_control_runner.py test --job-dir ./logs/tmp --params params/turtlebot/turtlebot_control_navigator_params.py -d 0
```
