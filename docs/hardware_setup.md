# Hardware and robot setup (what we found and what we changed)

Values read off our robot in Oct 2026. **We never changed robot-side ROS configuration** (domain, RMW, namespace, Create 3 settings): the lab's lidar/odometry setup depends on it.

## Robot

| Item | Value |
|---|---|
| Model | TurtleBot4 Standard, turtlebot4 software 2.0.2, ROS 2 Jazzy |
| ROS | `ROS_DOMAIN_ID=0`, `rmw_fastrtps_cpp`, simple discovery, no namespace |
| Camera | Luxonis OAK-D Pro (`03e7:f63b`) |
| Lidar | RPLIDAR (CP210x USB-UART); motor stops while docked |
| Odometry | Create 3, `odom → base_link` on `/tf` (no `/odom` topic on the Pi) |
| Camera extrinsic | `base_link → shell_link → oakd_camera_bracket → oakd_link → …optical`; optical centre ≈ 0.244 m above the floor |

## Camera: stock config has no depth

The stock bringup runs the OAK-D in `RGB` mode (250×250 preview). We run a **second depthai node `/oakd_rgbd`** at runtime (`robot/oakd_rgbd.yaml`), after `/oakd/stop_camera` releases the device. Nothing under `/etc` or `/opt` is modified; a reboot restores the stock camera.

- RGB `bgr8` 640×360 and depth `16UC1` 640×360 (mm), both 15 Hz, depth aligned to RGB, header stamps match to ~0.1 ms.
- Intrinsics: fx 514.01, fy 513.74, cx 324.53, cy 188.61; `rational_polynomial` distortion (8 coefficients; MonoGS supports only 5, so `bag_to_tum.py` undistorts).
- Depth scale: 1000 (mm). Check: target at 1219 mm read 1268 raw. Minimum range ≈ 0.7 m; untextured surfaces give no depth.

## Things that went wrong (so you don't repeat them)

- **Runtime switch of the stock node to RGBD** (`ros2 param set` + `stop_camera`): the camera container segfaulted. Use a separate node instead.
- **IR dot projector at 800 mA** browned out the robot's USB hub: OAK-D dropped, lidar driver hung, Create 3 stopped publishing. Pi reboot was not enough; a **full Create 3 power cycle** (hold power ~10 s off the dock, re-dock) fixed it. Keep `i_enable_ir: false`.
- `ros2 bag record` started as a background job in a non-interactive shell **ignores SIGINT**: use `set -m`, fall back to SIGTERM.
- The recorder can wait ~30 s for topics that never publish (`/cmd_vel`, `/hazard_detection`): wait only for the camera and `/tf` subscriptions.
- Stereo depth reads **shiny-floor reflections as near obstacles**: clear-ahead checks use only the upper half of the centre patch.
- The Create 3 **backup safety limit** blocks long reverse drives.
- `pkill -f '<pattern>'` over ssh kills the ssh shell whose command line contains the pattern: use `'[r]os2 bag'`-style patterns.

## Workstation (WSL2 on Windows 11, RTX 5080)

- WSL2 has **no CUDA IPC**: MonoGS's frontend/backend processes fail with "invalid resource handle". Run the backend as a thread (`pipeline/patch_monogs_threads.py`).
- Blackwell (sm_120) needs CUDA ≥ 12.8; build the rasterizers with `TORCH_CUDA_ARCH_LIST=12.0` and add `#include <cfloat>` / `<cstdint>` (see `pipeline/install_monogs.sh`).
- Evaluation needs `evo==1.11.0` and `matplotlib==3.5.3`; set `WANDB_MODE=disabled` (`--eval` turns wandb on).
- WSL DNS was flaky: set `dnsTunneling=false` in `.wslconfig`, and pre-download LPIPS/VGG weights before evaluation.
- Mirrored WSL networking does not cover a Windows Mobile Hotspot adapter, so the robot was driven and recorded over ssh, not via ROS from WSL.
- Standalone GLOMAP 1.2 cannot open a COLMAP 4.2 database; use `colmap global_mapper`.
