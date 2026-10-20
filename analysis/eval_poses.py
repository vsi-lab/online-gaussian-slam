"""Score camera trajectories from COLMAP / GLOMAP / MonoGS against wheel odometry, with one shared method.

Reference: <dataset>/pose.txt (odom -> camera optical, from the robot's wheel odometry; NOT ground truth).
Each estimate is aligned to the reference with Umeyama: Sim(3) (rotation, translation, scale) for all
methods, and SE(3) (no scale) too for MonoGS, which is metric because it uses depth. RGB-only SfM has no
metric scale, so only its Sim(3) number is meaningful.

Usage:
  python eval_poses.py --dataset <ds> [--colmap <sparse_txt_dir>] [--glomap <sparse_txt_dir>]
                       [--monogs <results/.../stamp>] --out <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def umeyama(src, dst, with_scale):
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if with_scale else 1.0
    return s, R, md - s * R @ ms


def ate(est, ref, with_scale):
    s, R, t = umeyama(est, ref, with_scale)
    al = (s * (R @ est.T)).T + t
    err = np.linalg.norm(al - ref, axis=1)
    return {"rmse_m": float(np.sqrt(np.mean(err**2))), "max_m": float(err.max()), "scale": float(s), "n": len(est)}


def ref_positions(ds):
    a = np.loadtxt(ds / "pose.txt")
    return {f"{t:.6f}": r[1:4] for t, r in zip(a[:, 0], a)}


def colmap_centres(txt_dir):
    """images.txt: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (world->cam), then a points line."""
    out = {}
    lines = [l for l in open(Path(txt_dir) / "images.txt") if not l.startswith("#")]
    for l in lines[0::2]:
        p = l.split()
        if len(p) < 10:
            continue
        q = [float(x) for x in p[1:5]]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        tvec = np.array([float(x) for x in p[5:8]])
        out[Path(p[9]).stem] = -R.T @ tvec  # camera centre in the SfM world
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--colmap", type=Path)
    ap.add_argument("--glomap", type=Path)
    ap.add_argument("--monogs", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ref = ref_positions(args.dataset)
    n_frames = len(ref)
    res = {"dataset": str(args.dataset), "frames": n_frames, "reference": "wheel odometry (not ground truth)"}
    for name, d in (("colmap", args.colmap), ("glomap", args.glomap)):
        if d is None:
            continue
        c = colmap_centres(d)
        keys = sorted(k for k in c if k in ref)
        est, gt = np.array([c[k] for k in keys]), np.array([ref[k] for k in keys])
        res[name] = {"registered": len(keys), "registered_frac": len(keys) / n_frames, "sim3": ate(est, gt, True)}
    if args.monogs:
        trj = json.load(open(args.monogs / "plot" / "trj_final.json"))
        est = np.array([np.array(p)[:3, 3] for p in trj["trj_est"]])
        gt = np.array([np.array(p)[:3, 3] for p in trj["trj_gt"]])
        res["monogs"] = {"keyframes": len(est), "se3": ate(est, gt, False), "sim3": ate(est, gt, True)}
    args.out.write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, json.dumps(v))


if __name__ == "__main__":
    main()
