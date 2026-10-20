"""Figures that show the Gaussians themselves (run in ~/MonoGS, monogs env, PYTHONPATH=.).

1. splats: one view rendered normally next to the same view with every Gaussian shrunk (scaling_modifier),
   so individual splats are visible. Uses a MonoGS result.
2. topdown: floor plan of a 3DGS model (e.g. GLOMAP + 3DGS on the L-route): Gaussian centres in a height
   slice, seen from above in their own colour, with the camera path. The SfM model is brought into the
   odometry frame with the Sim(3) fitted between its camera centres and the odometry camera positions.

Usage:
  python splat_viz.py splats  --result results/datasets_turtlebot/<stamp> --tag run_01 --out <dir> [--shrink 0.25]
  python splat_viz.py topdown --gs-model ~/sfm_compare/run_05_L/gs_glomap/model --sfm-txt ~/sfm_compare/run_05_L/glomap/txt \
                              --dataset datasets/turtlebot/run_05_L --tag run_05_L --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).parent))
from eval_poses import colmap_centres, ref_positions, umeyama  # noqa: E402

C0 = 0.28209479177387814  # SH band-0 constant


def splats(args):
    import cv2
    import torch
    from munch import munchify

    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from make_figures import load_cfg, make_cam

    cfg = load_cfg(args.result)
    calib = cfg["Dataset"]["Calibration"]
    pipe = munchify(cfg["pipeline_params"])
    g = GaussianModel(0, config=cfg)
    g.load_ply(str(args.result / "point_cloud" / "final" / "point_cloud.ply"))
    bg = torch.tensor([0.08, 0.09, 0.12], device="cuda")
    est = [np.array(p) for p in json.load(open(args.result / "plot" / "trj_final.json"))["trj_est"]]
    cam = make_cam(est[len(est) // 2], calib)
    outs = []
    with torch.no_grad():
        for mod in (1.0, args.shrink):
            img = render(cam, g, pipe, bg, scaling_modifier=mod)["render"].clamp(0, 1)
            outs.append(np.ascontiguousarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ("rendered Gaussian map", f"{g.get_xyz.shape[0]:,} Gaussians at {int(args.shrink * 100)}% size")
    for o, t in zip(outs, labels):
        cv2.putText(o, t, (12, 30), font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(o, t, (12, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    Image.fromarray(np.concatenate(outs, 1)).save(args.out / f"{args.tag}_splats.png", dpi=(300, 300))
    print(f"saved {args.tag}_splats.png ({g.get_xyz.shape[0]} Gaussians)")


def topdown(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    it = sorted((args.gs_model / "point_cloud").glob("iteration_*"), key=lambda p: int(p.name.split("_")[1]))[-1]
    v = PlyData.read(str(it / "point_cloud.ply"))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    rgb = np.clip(0.5 + C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"])))

    # Sim(3) SfM -> odometry frame from camera centres
    ref = ref_positions(args.dataset)
    cc = colmap_centres(args.sfm_txt)
    ks = sorted(k for k in cc if k in ref)
    s, R, t = umeyama(np.array([cc[k] for k in ks]), np.array([ref[k] for k in ks]), True)
    P = (s * (R @ xyz.T)).T + t
    cam = np.array([ref[k] for k in sorted(ref)])
    zc = np.median(cam[:, 2])  # camera height in odom frame
    keep = (op > 0.5) & (P[:, 2] > zc - 0.20) & (P[:, 2] < zc + 0.9)  # slice: just below camera to ~1.1 m above floor
    lo = cam[:, :2].min(0) - np.array([0.6, 2.2])
    hi = cam[:, :2].max(0) + np.array([3.2, 1.2])
    keep &= np.all((P[:, :2] > lo) & (P[:, :2] < hi), 1)
    order = np.argsort(P[keep, 2])  # draw low first, so higher structure sits on top

    fig, ax = plt.subplots(figsize=(args.size, args.size * (hi[1] - lo[1]) / (hi[0] - lo[0])), dpi=300)
    ax.set_facecolor("#F4F2EC")
    ax.scatter(P[keep][order, 0], P[keep][order, 1], c=rgb[keep][order], s=args.ms, linewidths=0, alpha=0.85)
    ax.plot(cam[:, 0], cam[:, 1], color="#1F2A4D", lw=3, ls="--", label="robot camera path (odometry)")
    ax.plot(*cam[0, :2], "o", color="#1F2A4D", ms=11)
    ax.annotate("start", cam[0, :2], xytext=(10, 10), textcoords="offset points", color="#1F2A4D", fontsize=args.fs, weight="bold")
    ax.annotate("end", cam[-1, :2], xytext=(10, 0), textcoords="offset points", color="#1F2A4D", fontsize=args.fs, weight="bold")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=args.fs)
    ax.set_ylabel("y (m)", fontsize=args.fs)
    ax.tick_params(labelsize=args.fs - 2)
    ax.legend(loc="lower left", fontsize=args.fs - 2, facecolor="#FFFFFF", edgecolor="#8A8F9C")
    fig.tight_layout()
    fig.savefig(args.out / f"{args.tag}_topdown.png", facecolor="white")
    print(f"saved {args.tag}_topdown.png: {keep.sum():,} of {len(P):,} Gaussians in slice, scale {s:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["splats", "topdown"])
    ap.add_argument("--result", type=Path)
    ap.add_argument("--gs-model", type=Path)
    ap.add_argument("--sfm-txt", type=Path)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shrink", type=float, default=0.25)
    ap.add_argument("--size", type=float, default=9.3, help="figure width in inches (poster column)")
    ap.add_argument("--fs", type=int, default=16)
    ap.add_argument("--ms", type=float, default=3.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splats(args) if args.mode == "splats" else topdown(args)


if __name__ == "__main__":
    main()
