"""Print the median raw depth in a centre patch over N frames (run on the robot).

Usage: python3 depth_center.py [n_frames] [patch]
"""
import sys

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
P = int(sys.argv[2]) if len(sys.argv) > 2 else 20
vals = []


def cb(msg):
    d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width]
    h, w = d.shape
    # upper half of the centre patch only: below the horizon the camera sees the (shiny) floor,
    # which stereo misreads as near
    c = d[h // 2 - P // 2 : h // 2, w // 2 - P // 2 : w // 2 + P // 2]
    c = c[c > 0]
    if c.size:
        vals.append((np.median(c), c.size / (P * P / 2), np.percentile(c, 10), np.percentile(c, 90)))


rclpy.init()
node = rclpy.create_node("depth_center")
node.create_subscription(Image, "/oakd_rgbd/stereo/image_raw", cb, qos_profile_sensor_data)
while rclpy.ok() and len(vals) < N:
    rclpy.spin_once(node, timeout_sec=1.0)
v = np.array(vals)
print(f"frames {len(v)}  centre {P}x{P} patch: median raw {np.median(v[:, 0]):.0f} "
      f"(frame-to-frame std {np.std(v[:, 0]):.1f}), valid {np.mean(v[:, 1]) * 100:.0f}%, "
      f"p10-p90 {np.median(v[:, 2]):.0f}-{np.median(v[:, 3]):.0f}")
node.destroy_node()
rclpy.shutdown()
