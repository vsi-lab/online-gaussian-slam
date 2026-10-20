"""Grab one RGB + depth frame on the robot and save a PPM with near pixels (< NEAR m) tinted red.

Usage (on the robot): python3 snap_near.py [near_m] [out.ppm]
"""
import sys

import numpy as np
import rclpy
from sensor_msgs.msg import Image

NEAR = float(sys.argv[1]) if len(sys.argv) > 1 else 2.9
OUT = sys.argv[2] if len(sys.argv) > 2 else "snap.ppm"
got = {}

rclpy.init()
node = rclpy.create_node("snap_near")
node.create_subscription(Image, "/oakd_rgbd/rgb/image_raw", lambda m: got.setdefault("rgb", m), 5)
node.create_subscription(Image, "/oakd_rgbd/stereo/image_raw", lambda m: got.setdefault("d", m), 5)
while rclpy.ok() and len(got) < 2:
    rclpy.spin_once(node, timeout_sec=1.0)

m = got["rgb"]
rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, : m.width * 3].reshape(m.height, m.width, 3)[..., ::-1].copy()
dm = got["d"]
d = np.frombuffer(dm.data, np.uint16).reshape(dm.height, dm.step // 2)[:, : dm.width] / 1000.0
near = (d > 0.2) & (d < NEAR)
rgb[near] = (0.4 * rgb[near] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
h, w = d.shape
rgb[h // 2 - 40 : h // 2 + 40, [w // 2 - 40, w // 2 + 40]] = 255  # the 80x80 check box
rgb[[h // 2 - 40, h // 2 + 40], w // 2 - 40 : w // 2 + 40] = 255
c = d[h // 2 - 40 : h // 2 + 40, w // 2 - 40 : w // 2 + 40]
cn = (c > 0.2) & (c < NEAR)
if cn.any():
    ys, xs = np.nonzero(cn)
    print(f"near pixels in check box: {cn.sum()}, rows {ys.min()}-{ys.max()}, cols {xs.min()}-{xs.max()} "
          f"(box is 80x80), depth {c[cn].min():.2f}-{c[cn].max():.2f} m")
else:
    print("no near pixels in check box")
open(OUT, "wb").write(b"P6 %d %d 255\n" % (w, h) + rgb.tobytes())
node.destroy_node()
rclpy.shutdown()
