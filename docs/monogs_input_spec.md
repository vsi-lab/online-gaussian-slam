# What MonoGS (RGB-D, TUM mode) expects

Read from `~/MonoGS` (commit 6c9254c): `utils/dataset.py` (`TUMParser`, `MonocularDataset`), `utils/slam_frontend.py`, `configs/rgbd/tum/*.yaml`. 2026-10-09.

## Folder layout

```
<dataset_path>/
  rgb.txt            # "timestamp filename" per line, e.g. "1305031452.791720 rgb/1305031452.791720.png"
  depth.txt          # same format, e.g. "... depth/1305031452.791720.png"
  groundtruth.txt    # OR pose.txt. REQUIRED, the loader crashes without one
  rgb/*.png
  depth/*.png
```

- `rgb.txt` / `depth.txt` are parsed with `np.loadtxt(delimiter=" ")`, with **no header skip**. TUM's own files start with `#` comment lines, which loadtxt skips. Our files: no header, or `#` comments only.
- `groundtruth.txt` / `pose.txt`: **first row is skipped** (`skiprows=1`), so write one `#` header line. Columns: `timestamp tx ty tz qx qy qz qw` (TUM convention; the loader rolls to `qw qx qy qz` internally). Pose is camera-to-world. MonoGS inverts it to get world-to-camera.
- Timestamps are float seconds. Any consistent clock works (the robot's wrong clock is fine as long as all three files use it).

## Frame association

- For each RGB timestamp, take the nearest depth and nearest pose. **Keep the frame only if both are within 0.08 s.** So the pose file must cover the whole sequence densely (TF odom at ~30 Hz is fine).
- Then frames are subsampled so consecutive kept frames are more than 1/32 s apart (`frame_rate=32`). At 15 fps, everything is kept.

## Images

- RGB: 8-bit, 3-channel PNG, read with PIL. **RGB order** (cv2 users: convert from BGR before writing).
- Depth: **16-bit single-channel PNG**, metres = `pixel / depth_scale`. 0 = invalid.
- Depth must be **pixel-aligned to the RGB image** (same size, same intrinsics).

## Config (`Dataset.Calibration`)

```yaml
inherit_from: "configs/rgbd/tum/base_config.yaml"
Dataset:
  dataset_path: "datasets/turtlebot/<seq>"
  Calibration:
    fx: ...   fy: ...   cx: ...   cy: ...
    k1: 0  k2: 0  p1: 0  p2: 0  k3: 0
    distorted: False
    width: 640
    height: 360
    depth_scale: 1000.0     # TUM uses 5000; depthai 16UC1 is mm, so 1000 (VERIFY with tape measure)
```

- Distortion support is **5-coefficient plumb_bob only** (`k1 k2 p1 p2 k3`, applied with `cv2.remap`). The OAK-D publishes **`rational_polynomial` (8 coefficients)**, which MonoGS can't use directly. **The converter must undistort RGB itself (cv2 with all 8 coefs) and write `distorted: False`.** The depth aligned to RGB is already in the rectified RGB frame; check this once we have a frame.
- `depth_scale` present means depth is used (`has_depth`). Missing means monocular.

## How poses are used (honesty check for the poster)

- `slam_frontend.initialize()` places **only the first frame** at the provided pose (`update_RT(R_gt, T_gt)`), which just fixes the world frame.
- All later poses are **estimated by MonoGS** (photometric + depth tracking). The provided poses are otherwise used only for **evaluation** (ATE in `--eval`).
- So feeding TF odometry as `pose.txt` keeps this as real SLAM, with odometry as the yardstick, as section 9 of the context doc requires.

## Running

```bash
conda activate monogs
cd ~/MonoGS
python slam.py --config <cfg>.yaml --eval   # headless: save_results, no GUI, eval_rendering → ATE + PSNR/SSIM/LPIPS
```

Results go to `results/<dataset>/<timestamp>/` (ply, trajectory plots, metrics json).

## Environment changes vs upstream (RTX 5080, sm_120)

Upstream: Python 3.7, PyTorch 1.12.1, CUDA 11.6, which doesn't support Blackwell. Our env `monogs` (`tools/install_monogs.sh`):

- Python 3.10, PyTorch 2.7.1+cu128, torchvision 0.22.1
- CUDA 12.8 nvcc/libs from conda (`nvidia/label/cuda-12.8.1`), `CUDA_HOME=$CONDA_PREFIX`
- `TORCH_CUDA_ARCH_LIST=12.0`
- `numpy<2`, open3d 0.18.0, opencv-python 4.10.0.84, latest evo
- Source patches: `#include <cfloat>` in `simple-knn/simple_knn.cu`, `#include <cstdint>` in `diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`
