"""Cross-run test: how well does one run's Gaussian map predict a LATER drive's camera images?

Map run (e.g. run_01) gives a MonoGS map in its world frame. A test run (run_02 = same path, run_03 = shifted
one tile right) is placed in that frame by ICP of its depth (first frames, chained with the test run's own
odometry) against the map's Gaussian centres, starting from --init-right metres to the right of the map run's
first keyframe. The robot was carried between runs, so wheel odometry cannot relate the runs.
Then every test frame is rendered from the map and compared with the real image.

Usage (in ~/MonoGS, monogs env, PYTHONPATH=.):
  python cross_run_eval.py --map-result results/datasets_turtlebot/<stamp> --test datasets/turtlebot/run_03 \
      --start 2.7 --end 22.1 --init-right 0.305 --tag run_03 --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from munch import munchify
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from make_figures import load_cfg, make_cam, offset_pose, render_rgb  # noqa: E402

from gaussian_splatting.gaussian_renderer import render  # noqa: E402
from gaussian_splatting.scene.gaussian_model import GaussianModel  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from utils.pose_utils import update_pose  # noqa: E402
from utils.slam_utils import get_loss_tracking  # noqa: E402


def localize(cam, real_rgb, depth_m, gauss, pipe, bg, cfg, iters):
    """MonoGS-style tracking of one frame against a FROZEN map (pose + exposure only)."""
    cam.original_image = torch.from_numpy(real_rgb / 255.0).float().permute(2, 0, 1).cuda()
    cam.depth = depth_m
    cam.compute_grad_mask(cfg)
    lr = cfg["Training"]["lr"]
    opt = torch.optim.Adam([
        {"params": [cam.cam_rot_delta], "lr": lr["cam_rot_delta"]},
        {"params": [cam.cam_trans_delta], "lr": lr["cam_trans_delta"]},
        {"params": [cam.exposure_a], "lr": 0.01},
        {"params": [cam.exposure_b], "lr": 0.01},
    ])
    for _ in range(iters):
        pkg = render(cam, gauss, pipe, bg)
        opt.zero_grad()
        loss = get_loss_tracking(cfg, pkg["render"], pkg["depth"], pkg["opacity"], cam)
        loss.backward()
        with torch.no_grad():
            opt.step()
            if update_pose(cam):
                break
    with torch.no_grad():
        img = render(cam, gauss, pipe, bg)["render"]
        img_ab = torch.exp(cam.exposure_a) * img + cam.exposure_b
    to8 = lambda t: (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    w2c = np.eye(4)
    w2c[:3, :3] = cam.R.detach().cpu().numpy()
    w2c[:3, 3] = cam.T.detach().cpu().numpy()
    return np.linalg.inv(w2c), to8(img), to8(img_ab)


def tum_to_mats(path):
    a = np.loadtxt(path)
    Ts = []
    for r in a:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(r[4:8]).as_matrix()
        T[:3, 3] = r[1:4]
        Ts.append(T)
    return a[:, 0], Ts


def depth_points(depth_png, calib, stride=4, max_d=4.0):
    d = np.array(Image.open(depth_png)).astype(np.float32) / calib["depth_scale"]
    H, W = d.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = d[::stride, ::stride]
    m = (z > 0.3) & (z < max_d)
    x = (u[m] - calib["cx"]) * z[m] / calib["fx"]
    y = (v[m] - calib["cy"]) * z[m] / calib["fy"]
    return np.stack([x, y, z[m]], 1)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 10 * np.log10(255.0**2 / max(mse, 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-result", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True, help="converted test-run dataset dir")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--init-right", type=float, default=0.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.map_result)
    cfg["Training"]["monocular"] = False  # set at runtime by slam.py, not saved in config.yml
    calib = cfg["Dataset"]["Calibration"]
    pipe = munchify(cfg["pipeline_params"])
    gauss = GaussianModel(0, config=cfg)
    gauss.load_ply(str(args.map_result / "point_cloud" / "final" / "point_cloud.ply"))
    bg = torch.zeros(3, device="cuda")
    est = [np.array(p) for p in json.load(open(args.map_result / "plot" / "trj_final.json"))["trj_est"]]

    # test run: frames + its own odometry (camera-to-odom), relative to its first frame
    ts, P = tum_to_mats(args.test / "pose.txt")
    rgb = [l.split()[1] for l in open(args.test / "rgb.txt") if l.strip()]
    dep = [l.split()[1] for l in open(args.test / "depth.txt") if l.strip()]
    keep = [i for i, t in enumerate(ts) if args.start <= t - ts[0] <= args.end]
    i0 = keep[0]
    rel = [np.linalg.inv(P[i0]) @ P[i] for i in range(len(P))]  # cam_i0 <- cam_i

    # source cloud: first ~2 s of the clip (robot still), in the test run's first-camera frame
    src = np.concatenate([(rel[i][:3, :3] @ depth_points(args.test / dep[i], calib).T).T + rel[i][:3, 3]
                          for i in keep[:30:3]])
    tgt = gauss.get_xyz.detach().cpu().numpy()
    s_pc, t_pc = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    s_pc.points = o3d.utility.Vector3dVector(src)
    t_pc.points = o3d.utility.Vector3dVector(tgt)
    s_pc = s_pc.voxel_down_sample(0.02)
    t_pc = t_pc.voxel_down_sample(0.02)
    t_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))

    T = offset_pose(est[0], right=args.init_right)  # initial guess: world <- test cam_i0
    for thr in (0.20, 0.08, 0.04):
        reg = o3d.pipelines.registration.registration_icp(
            s_pc, t_pc, thr, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T = reg.transformation
    d0 = np.linalg.inv(est[0]) @ T  # test start expressed in the map run's first camera frame
    print(f"ICP fitness {reg.fitness:.3f}, inlier RMSE {reg.inlier_rmse * 100:.1f} cm")
    print(f"test start relative to map start (camera frame x right, y down, z fwd): "
          f"{np.round(d0[:3, 3], 3).tolist()} m, yaw {np.degrees(np.arctan2(d0[0, 2], d0[2, 2])):.1f} deg")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows, samples = [], []
    c2w_prev, i_prev = T, i0
    for n, i in enumerate(keep):
        guess = c2w_prev @ np.linalg.inv(rel[i_prev]) @ rel[i]  # previous tracked pose + odometry step
        real = np.array(Image.open(args.test / rgb[i]).convert("RGB"))
        dm = np.array(Image.open(args.test / dep[i])).astype(np.float32) / calib["depth_scale"]
        cam = make_cam(guess, calib)
        c2w, r_raw, r_ab = localize(cam, real, dm, gauss, pipe, bg, cfg, 300 if n == 0 else 100)
        c2w_prev, i_prev = c2w, i
        lat = (np.linalg.inv(est[0]) @ c2w)[:3, 3]
        rows.append((ts[i] - ts[i0], psnr(real, r_raw), psnr(real, r_ab), lat[0], lat[2]))
        if n in (len(keep) // 5, len(keep) // 2, 4 * len(keep) // 5):
            samples.append(np.concatenate([real, r_ab], 1))
    R = np.array(rows)
    p, pab = R[:, 1], R[:, 2]
    np.savetxt(out / f"{args.tag}_psnr.csv", R, delimiter=",",
               header="t_s,psnr_raw_db,psnr_exposure_corrected_db,x_right_of_map_start_m,z_fwd_of_map_start_m", fmt="%.3f")
    Image.fromarray(np.concatenate(samples, 0)).save(out / f"{args.tag}_cross_samples.png")
    summary = {"tag": args.tag, "map": str(args.map_result), "test": str(args.test), "frames": len(p),
               "psnr_raw_mean": float(p.mean()), "psnr_ab_mean": float(pab.mean()), "psnr_ab_median": float(np.median(pab)),
               "psnr_ab_p10": float(np.percentile(pab, 10)), "lateral_offset_median_m": float(np.median(R[:, 3])),
               "icp_fitness": float(reg.fitness), "icp_rmse_m": float(reg.inlier_rmse),
               "start_offset_cam_m": d0[:3, 3].round(3).tolist()}
    (out / f"{args.tag}_cross.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.tag}: {len(p)} frames; PSNR raw {p.mean():.2f} dB, exposure-corrected {pab.mean():.2f} dB "
          f"(median {np.median(pab):.2f}, p10 {np.percentile(pab, 10):.2f}); tracked lateral offset median {np.median(R[:, 3]) * 100:.1f} cm")


if __name__ == "__main__":
    main()
