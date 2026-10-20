"""Poster figures from a finished MonoGS run (run inside ~/MonoGS with the monogs env).

Produces in --out:
  <tag>_real_vs_render.png   real frame | rendered from MonoGS's estimated pose | abs error   (fig 3)
  <tag>_novel_view.png       render from a viewpoint the robot never occupied             (fig 2)
  <tag>_trajectory.png       top-down: MonoGS keyframes (aligned) vs wheel odometry       (fig 4)
  <tag>_flythrough.mp4       camera sweeping along the path, raised and pulled back
  <tag>_figures.txt          provenance

Usage:
  python make_figures.py --result results/datasets_turtlebot/<stamp> --tag run_01 --out /mnt/c/.../poster/figures
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from munch import munchify
from PIL import Image

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.camera_utils import Camera

DPI = 300


def load_cfg(result):
    cfg = yaml.safe_load(open(result / "config.yml"))
    return cfg


def make_cam(c2w, calib):
    W, H = calib["width"], calib["height"]
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    proj = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H).transpose(0, 1)
    fovx = 2 * np.arctan(W / (2 * fx))
    fovy = 2 * np.arctan(H / (2 * fy))
    cam = Camera(0, None, None, torch.eye(4), proj, fx, fy, cx, cy, fovx, fovy, H, W)
    w2c = np.linalg.inv(c2w)
    cam.update_RT(torch.tensor(w2c[:3, :3], dtype=torch.float32), torch.tensor(w2c[:3, 3], dtype=torch.float32))
    return cam


@torch.no_grad()
def render_rgb(gauss, cam, pipe, bg):
    out = render(cam, gauss, pipe, bg)
    img = out["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def offset_pose(c2w, up=0.0, back=0.0, right=0.0, pitch_deg=0.0, yaw_deg=0.0):
    """Move/rotate a camera in its own optical frame (x right, y down, z forward)."""
    T = np.eye(4)
    T[:3, 3] = [right, -up, -back]
    p, y = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])  # +pitch = look down (y is down)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    T[:3, :3] = Ry @ Rx
    return c2w @ T


def umeyama_se3(src, dst):
    """Rigid (no scale) alignment src -> dst, both Nx3."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    return R, md - R @ ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kf", type=int, default=None, help="keyframe index for real-vs-render (default: middle)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.result)
    calib = cfg["Dataset"]["Calibration"]
    ds = Path(cfg["Dataset"]["dataset_path"])
    pipe = munchify(cfg["pipeline_params"])
    sh = cfg["model_params"]["sh_degree"] if not cfg["Training"].get("spherical_harmonics") is False else 0

    gauss = GaussianModel(sh, config=cfg)
    gauss.load_ply(str(args.result / "point_cloud" / "final" / "point_cloud.ply"))
    gauss.active_sh_degree = gauss.max_sh_degree
    bg = torch.zeros(3, device="cuda")
    print(f"gaussians: {gauss.get_xyz.shape[0]}")

    trj = json.load(open(args.result / "plot" / "trj_final.json"))
    ids = trj["trj_id"]
    est = [np.array(p) for p in trj["trj_est"]]  # camera-to-world (MonoGS world)
    gt = [np.array(p) for p in trj["trj_gt"]]

    rgb_files = [l.split()[1] for l in open(ds / "rgb.txt") if l.strip() and not l.startswith("#")]
    W, H = calib["width"], calib["height"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    def label(img, text):
        img = img.copy()
        cv2.putText(img, text, (10, 28), font, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (10, 28), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    # --- fig 3: real vs rendered from the same (estimated) pose
    k = args.kf if args.kf is not None else len(ids) // 2
    real = np.array(Image.open(ds / rgb_files[ids[k]]).convert("RGB"))
    rend = render_rgb(gauss, make_cam(est[k], calib), pipe, bg)
    err = np.abs(real.astype(np.float32) - rend.astype(np.float32)).mean(2)
    err_c = (plt.cm.inferno(np.clip(err / 60.0, 0, 1))[..., :3] * 255).astype(np.uint8)
    psnr = 10 * np.log10(255.0**2 / np.mean((real.astype(np.float32) - rend.astype(np.float32)) ** 2))
    strip = np.concatenate([label(real, "real camera frame"), label(rend, f"MonoGS render (PSNR {psnr:.1f} dB)"),
                            label(err_c, "|error|")], axis=1)
    Image.fromarray(strip).save(args.out / f"{args.tag}_real_vs_render.png", dpi=(DPI, DPI))

    # --- fig 2: novel view inside the observed space: from the first keyframe, 0.6 m forward, raised 0.35 m,
    # 0.3 m to the right, looking 15 deg down and 12 deg left
    nv_pose = offset_pose(est[0], up=0.35, back=-0.6, right=0.3, pitch_deg=15, yaw_deg=-12)
    nv = render_rgb(gauss, make_cam(nv_pose, calib), pipe, bg)
    Image.fromarray(label(nv, "novel view: 0.35 m above, 0.3 m right of robot path")).save(
        args.out / f"{args.tag}_novel_view.png", dpi=(DPI, DPI))

    # --- fig 4: trajectory, top-down in the odometry frame
    P_est = np.array([p[:3, 3] for p in est])
    P_gt = np.array([p[:3, 3] for p in gt])
    R, t = umeyama_se3(P_est, P_gt)
    P_al = (R @ P_est.T).T + t
    ate = np.sqrt(np.mean(np.sum((P_al - P_gt) ** 2, 1)))
    odom = np.loadtxt(ds / "pose.txt")[:, 1:4]
    span = np.ptp(np.vstack([odom[:, :2], P_al[:, :2]]), 0)
    fig, ax = plt.subplots(figsize=(6.5, 2.8) if span[1] < 0.9 else (6.0, 5.0), dpi=DPI)
    ax.plot(odom[:, 0], odom[:, 1], color="#888888", lw=2, label="wheel odometry (all frames)")
    ax.plot(P_gt[:, 0], P_gt[:, 1], "o", color="#888888", ms=4, mfc="white")
    ax.plot(P_al[:, 0], P_al[:, 1], "-o", color="#1f6fd1", lw=1.6, ms=4, label="MonoGS estimate (keyframes)")
    ax.set_aspect("equal")
    allp = np.vstack([odom[:, :2], P_al[:, :2]])
    lo, hi = allp.min(0), allp.max(0)
    pad = 0.15
    if hi[1] - lo[1] < 0.9:  # near-straight path: show a sensible band instead of a hairline
        yc = (hi[1] + lo[1]) / 2
        lo[1], hi[1] = yc - 0.45, yc + 0.45
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{args.tag}: MonoGS vs wheel odometry, ATE RMSE {ate * 100:.1f} cm ({len(ids)} keyframes)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.out / f"{args.tag}_trajectory.png", dpi=DPI)
    plt.close(fig)

    # --- fly-through video: follow the keyframes, raised and pulled back, smooth interpolation
    frames = []
    # robot height with a gentle sideways sway (the offset sweep shows lateral moves hold up, raised ones don't)
    n_total = (len(est) - 1) * 12
    for i in range(len(est) - 1):
        for s in np.linspace(0, 1, 12, endpoint=False):
            c2w = est[i].copy()
            c2w[:3, 3] = (1 - s) * est[i][:3, 3] + s * est[i + 1][:3, 3]
            ph = 2 * np.pi * len(frames) / n_total
            pose = offset_pose(c2w, right=0.15 * np.sin(ph), yaw_deg=-6 * np.sin(ph))
            frames.append(render_rgb(gauss, make_cam(pose, calib), pipe, bg))
    vw = cv2.VideoWriter(str(args.out / f"{args.tag}_flythrough.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    (args.out / f"{args.tag}_figures.txt").write_text(
        f"source run: {args.result}\ndataset: {ds}\nkeyframes: {len(ids)}; real-vs-render keyframe {k} (frame {ids[k]}), "
        f"single-frame PSNR {psnr:.2f} dB\nnovel view: first keyframe +0.6 m fwd, +0.35 m up, +0.3 m right, 15 deg down, 12 deg left\n"
        f"trajectory ATE (keyframes, SE3-aligned): {ate * 100:.2f} cm\nflythrough: {len(frames)} frames @24 fps, robot height, +/-0.15 m sideways sway\n"
        f"DATA: real TurtleBot4 OAK-D Pro RGB-D (not TUM, not sim)\n")
    print(f"done: {args.tag} (kf {k}, frame PSNR {psnr:.1f} dB, ATE {ate * 100:.1f} cm, {len(frames)} video frames)")


if __name__ == "__main__":
    main()
