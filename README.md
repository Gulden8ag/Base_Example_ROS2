# Robot Arm Simulation - ROS2 Jazzy + Gazebo

## Requirements
```bash
sudo apt install ros-jazzy-gz-ros2-control ros-jazzy-ros2-controllers ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge ros-jazzy-robot-state-publisher ros-jazzy-rviz2 ros-jazzy-xacro
pip install numpy
```

## Build
```bash
colcon build
source install/setup.bash
```

## Launch
```bash
ros2 launch my_robot_bringup ros_gazebo.launch.xml
```

## Send a pose command
```bash
ros2 service call /ee_pose my_robot_interfaces/srv/EEpose "{x: 0.3, y: 0.2, z: 0.5}"
```
