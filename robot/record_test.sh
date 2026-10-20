# Save camera_info and record a short stationary test bag on the robot (run via: ssh ... bash -s)
source /opt/ros/jazzy/setup.bash
mkdir -p ~/turtlebot-gs/bags && cd ~/turtlebot-gs
timeout 10 ros2 topic echo --once /oakd_rgbd/rgb/camera_info > camera_info_rgb.yaml
timeout 10 ros2 topic echo --once /oakd_rgbd/stereo/camera_info > camera_info_stereo.yaml
grep -E "distortion_model|^d:|^k:|^width|^height" -A0 camera_info_rgb.yaml
rm -rf bags/test_static
timeout -s INT 20 ros2 bag record -s mcap -o bags/test_static \
  /oakd_rgbd/rgb/image_raw /oakd_rgbd/stereo/image_raw /oakd_rgbd/rgb/camera_info \
  /tf /tf_static /imu /scan > /dev/null 2>&1
ros2 bag info bags/test_static | grep -E "Duration|Size|Messages|Topic:"
df -h ~ | tail -1
