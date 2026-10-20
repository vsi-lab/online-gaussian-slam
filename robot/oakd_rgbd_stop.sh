# Stop our RGB-D camera node and give the device back to the stock /oakd node (run via: ssh ... bash -s)
source /opt/ros/jazzy/setup.bash
pkill -f "camera.launch.py name:=oakd_rgbd" ; pkill -f "__node:=oakd_rgbd_container" ; pkill -f "oakd_rgbd_state_publisher"
sleep 3
ps aux | grep -E "oakd_rgbd" | grep -v grep | wc -l
