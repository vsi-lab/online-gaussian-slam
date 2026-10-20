# Check our RGB-D camera node on the robot (run via: ssh ... bash -s)
source /opt/ros/jazzy/setup.bash
for i in $(seq 1 24); do
  timeout 5 ros2 topic list 2>/dev/null | grep -q '^/oakd_rgbd/stereo/image_raw$' && break
  grep -q -i -E "died|fatal|terminate" ~/turtlebot-gs/oakd_rgbd.log && break
  sleep 5
done
echo "=== log"; grep -v "^\s*$" ~/turtlebot-gs/oakd_rgbd.log | grep -i -E "error|warn|ready|died|pipeline|speed|connected|fatal" | tail -15
echo "=== stock container alive?"; ps aux | grep "__node:=oakd_container" | grep -v grep | wc -l
echo "=== topics"; timeout 10 ros2 topic list | grep oakd_rgbd
echo "=== rates"
for t in /oakd_rgbd/rgb/image_raw /oakd_rgbd/stereo/image_raw; do
  echo "$t: $(timeout 12 ros2 topic hz $t 2>&1 | grep average | head -1)"
done
echo "=== encodings / sizes"
for t in /oakd_rgbd/rgb/image_raw /oakd_rgbd/stereo/image_raw; do
  echo "$t: $(timeout 10 ros2 topic echo --once $t --field encoding 2>/dev/null | head -1) $(timeout 10 ros2 topic echo --once $t --field width 2>/dev/null | head -1)x$(timeout 10 ros2 topic echo --once $t --field height 2>/dev/null | head -1) frame=$(timeout 10 ros2 topic echo --once $t --field header.frame_id 2>/dev/null | head -1)"
done
echo "=== cpu"; top -bn1 | head -12 | tail -6
