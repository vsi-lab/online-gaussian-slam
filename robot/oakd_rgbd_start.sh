# Start our RGB-D camera node on the robot (run on the robot via: ssh ... bash -s).
# Runtime only: nothing under /etc or /opt is changed. A reboot restores the stock camera.
source /opt/ros/jazzy/setup.bash
cd ~/turtlebot-gs

# Release the device from the stock node (same call turtlebot4_node makes when docking)
timeout 30 ros2 service call /oakd/stop_camera std_srvs/srv/Trigger | tail -1
sleep 3

# Launch ours detached, logging to ~/turtlebot-gs/oakd_rgbd.log
setsid nohup ros2 launch depthai_ros_driver camera.launch.py \
  name:=oakd_rgbd parent_frame:=oakd_link camera_model:=OAK-D-PRO \
  params_file:=$HOME/turtlebot-gs/oakd_rgbd.yaml \
  > ~/turtlebot-gs/oakd_rgbd.log 2>&1 < /dev/null &
echo "launched pid $!"
