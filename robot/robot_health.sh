# Robot health check (run via: ssh ... bash -s). Waits for bringup, then reports
# the nodes and topic rates that the lab's lidar/odometry setup depends on.
source /opt/ros/jazzy/setup.bash
for i in $(seq 1 30); do
  timeout 5 ros2 node list --no-daemon 2>/dev/null | grep -q '^/rplidar_composition$' && break
  sleep 5
done
echo "=== nodes"; timeout 10 ros2 node list --no-daemon | grep -E 'rplidar|create3|oakd|turtlebot4_node|robot_state' | tr '\n' ' '; echo
echo "=== rates"
for t in /scan /tf /imu /oakd/rgb/preview/image_raw /oakd_rgbd/rgb/image_raw /oakd_rgbd/stereo/image_raw; do
  r=$(timeout 10 ros2 topic hz "$t" 2>&1 | grep average | head -1)
  echo "$t: ${r:-none}"
done
echo "=== usb"; lsusb | grep -E "10c4|03e7" | cut -c1-70
echo "=== dock"; timeout 8 ros2 topic echo --once /dock_status --field is_docked 2>/dev/null | head -1
