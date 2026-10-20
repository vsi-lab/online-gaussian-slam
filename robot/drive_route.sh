# Record a bag while driving: forward L1, rotate left 90 deg, forward L2 (run via: ssh ... bash -s).
# Before each forward leg the centre depth (10th percentile) must exceed the leg + 0.5 m, else abort.
source /opt/ros/jazzy/setup.bash
set -m   # job control: otherwise background jobs ignore SIGINT and the recorder never stops
NAME=${NAME:-run_04}; L1=${L1:-2.74}; L2=${L2:-3.05}; SPEED=${SPEED:-0.15}; WSPEED=${WSPEED:-0.25}
cd ~/turtlebot-gs/bags && rm -rf "$NAME"

clear_ahead() {  # $1 = metres needed
  p10=$(timeout 30 python3 ~/turtlebot-gs/depth_center.py 10 80 | sed -n 's/.*p10-p90 \([0-9]*\)-.*/\1/p')
  need=$(python3 -c "print(int(($1 + 0.5) * 1000))")
  echo "  clear-ahead check: p10 ${p10:-none} mm, need > $need mm"
  [ -n "$p10" ] && [ "$p10" -gt "$need" ]
}
stop_rec() {
  kill -INT $REC; for i in $(seq 1 20); do kill -0 $REC 2>/dev/null || break; sleep 1; done
  kill -0 $REC 2>/dev/null && { kill -TERM $REC; wait $REC; }
  ros2 bag info "$NAME" | grep -E "Duration|image_raw" | cut -c1-110
}
act() {  # $1 action, $2 type, $3 goal
  timeout 60 ros2 action send_goal "$1" "$2" "$3" 2>&1 | grep -E "Goal (accepted|rejected)|status" | tr '\n' ' '; echo
}

clear_ahead "$L1" || { echo "ABORT: obstacle within leg 1, robot did not move"; exit 1; }
ros2 bag record -s mcap -o "$NAME" --topics \
  /oakd_rgbd/rgb/image_raw /oakd_rgbd/stereo/image_raw /oakd_rgbd/rgb/camera_info \
  /tf /tf_static /imu /scan /dock_status > rec_$NAME.log 2>&1 &
REC=$!
for i in $(seq 1 30); do grep -q "Subscribed to topic '/oakd_rgbd/stereo/image_raw'" rec_$NAME.log && grep -q "Subscribed to topic '/tf'" rec_$NAME.log && break; sleep 1; done
echo "recording; settling 3 s"; sleep 3

echo "leg 1: forward $L1 m"; act /drive_distance irobot_create_msgs/action/DriveDistance "{distance: $L1, max_translation_speed: $SPEED}"
sleep 2
echo "rotate left 90 deg"; act /rotate_angle irobot_create_msgs/action/RotateAngle "{angle: 1.5708, max_rotation_speed: $WSPEED}"
sleep 2
if clear_ahead "$L2"; then
  echo "leg 2: forward $L2 m"; act /drive_distance irobot_create_msgs/action/DriveDistance "{distance: $L2, max_translation_speed: $SPEED}"
else
  echo "SKIPPED leg 2: obstacle within $L2 m after the turn"
fi
sleep 3
stop_rec
