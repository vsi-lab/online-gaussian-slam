# Record a bag on the robot while driving a straight out-and-back (run via: ssh ... bash -s).
# Usage: DIST=1.8 SPEED=0.15 NAME=run_01 ; robot must be on the floor with the strip ahead clear.
source /opt/ros/jazzy/setup.bash
set -m   # job control: otherwise background jobs ignore SIGINT and the recorder never stops
NAME=${NAME:-run_01}; DIST=${DIST:-1.8}; SPEED=${SPEED:-0.15}; BACK=${BACK:-0}  # BACK=1 tries to reverse (Create 3 backup limit blocked it on run_01)
cd ~/turtlebot-gs/bags && rm -rf "$NAME"

ros2 bag record -s mcap -o "$NAME" --topics \
  /oakd_rgbd/rgb/image_raw /oakd_rgbd/stereo/image_raw /oakd_rgbd/rgb/camera_info \
  /tf /tf_static /imu /scan /dock_status > rec_$NAME.log 2>&1 &
REC=$!
# wait until the camera topics are subscribed (/cmd_vel etc. never publish and stalled this for 30 s before)
for i in $(seq 1 30); do grep -q "Subscribed to topic '/oakd_rgbd/stereo/image_raw'" rec_$NAME.log && grep -q "Subscribed to topic '/tf'" rec_$NAME.log && break; sleep 1; done
echo "recording ($REC); settling 3 s"; sleep 3

drive() {  # $1 = signed distance
  timeout 40 ros2 action send_goal /drive_distance irobot_create_msgs/action/DriveDistance \
    "{distance: $1, max_translation_speed: $SPEED}" 2>&1 | grep -E "status|Goal (accepted|rejected)|SUCCEEDED|ABORTED|CANCELED" | tr '\n' ' '
  echo
}
echo "forward $DIST m"; drive "$DIST"
sleep 2
[ "$BACK" = 1 ] && { echo "back $DIST m"; drive "-$DIST"; }
sleep 3

kill -INT $REC; for i in $(seq 1 20); do kill -0 $REC 2>/dev/null || break; sleep 1; done
kill -0 $REC 2>/dev/null && { echo "recorder ignored SIGINT, sending TERM"; kill -TERM $REC; wait $REC; }
ros2 bag info "$NAME" | grep -E "Duration|Size|Count"
timeout 8 ros2 topic echo --once /hazard_detection --field detections 2>/dev/null | head -5
