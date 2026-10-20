"""Revisit planner (baseline): which parts of a Gaussian map should the robot go back and look at?

Given a finished MonoGS map and its keyframe poses (world = odometry frame, metric), for every Gaussian:
  * n_views  - keyframes in which it is in the frustum, within range, and not occluded (projected depth vs the
               rendered depth at that pixel)
  * spread   - angular spread of the directions it was seen from (max pairwise angle between view rays)
  * residual - mean |render - image| at its projected pixel over the keyframes that saw it
Per floor cell (x-y grid in the odometry frame, Gaussians in a height band) these are pooled (opacity-weighted)
into a priority score:  score = w_v * (1 - views_n) + w_a * (1 - spread_n) + w_r * residual_n   (each in [0, 1]).
For the top cells a revisit pose (x, y, yaw) is proposed on a ring around the cell, preferring a viewing
direction far from the existing ones (lateral views, since raised views are impossible for a ground robot) and
positions clear of mapped geometry.

This is a heuristic baseline (Prof. Lee's question; context doc section 12.4). Whether the score tracks true
reconstruction error must be checked against Isaac Sim ground truth.

Usage (in ~/MonoGS, monogs env, PYTHONPATH=.):
  python revisit_planner.py --result results/datasets_turtlebot/<stamp> --tag run_01 --out <dir> [--top 5]
Outputs: <tag>_revisit.png (heatmap + suggested poses), <tag>_revisit.json (goals + per-cell stats)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from munch import munchify
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from make_figures import load_cfg, make_cam  # noqa: E402

from gaussian_splatting.gaussian_renderer import render  # noqa: E402
from gaussian_splatting.scene.gaussian_model import GaussianModel  # noqa: E402


def visibility_pass(g, cams_c2w, images, calib, pipe, max_range, depth_tol):
    """Per-Gaussian view count, sum of unit view directions (for spread), list of view dirs, residual sums."""
    xyz = g.get_xyz.detach()
    N = xyz.shape[0]
    W, H = calib["width"], calib["height"]
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    bg = torch.zeros(3, device="cuda")
    n_views = torch.zeros(N, device="cuda")
    res_sum = torch.zeros(N, device="cuda")
    dirs = []  # per keyframe: (indices, unit directions camera->gaussian)
    for c2w, img in zip(cams_c2w, images):
        cam = make_cam(c2w, calib)
        with torch.no_grad():
            out = render(cam, g, pipe, bg)
        rdepth = out["depth"][0]
        rimg = out["render"].clamp(0, 1)
        real = torch.from_numpy(img / 255.0).float().permute(2, 0, 1).cuda()
        err = (rimg - real).abs().mean(0)  # H x W

        w2c = torch.from_numpy(np.linalg.inv(c2w)).float().cuda()
        pc = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        u = (fx * pc[:, 0] / z + cx).round().long()
        v = (fy * pc[:, 1] / z + cy).round().long()
        inside = (z > 0.2) & (z < max_range) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = torch.nonzero(inside).squeeze(1)
        uu, vv = u[idx], v[idx]
        d_r = rdepth[vv, uu]
        vis = (d_r > 0) & (z[idx] <= d_r * (1 + depth_tol) + 0.02)
        idx, uu, vv = idx[vis], uu[vis], vv[vis]
        n_views[idx] += 1
        res_sum[idx] += err[vv, uu]
        centre = torch.from_numpy(c2w[:3, 3]).float().cuda()
        d = xyz[idx] - centre
        dirs.append((idx.cpu().numpy(), (d / d.norm(dim=1, keepdim=True)).cpu().numpy()))
    return n_views.cpu().numpy(), res_sum.cpu().numpy(), dirs


def angular_spread(N, dirs):
    """Max pairwise angle (deg) between view directions per Gaussian, via the extremes of the 2D bearing."""
    lo = np.full(N, np.inf)
    hi = np.full(N, -np.inf)
    for idx, d in dirs:
        bearing = np.degrees(np.arctan2(d[:, 1], d[:, 0]))  # horizontal bearing in the odometry (world) frame
        np.minimum.at(lo, idx, bearing)
        np.maximum.at(hi, idx, bearing)
    spread = hi - lo
    spread[~np.isfinite(spread)] = 0
    return np.minimum(spread, 360 - spread)


def norm01(a):
    lo, hi = np.percentile(a, 5), np.percentile(a, 95)
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True, help="MonoGS results dir (config.yml, plot/, point_cloud/)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cell", type=float, default=0.25, help="floor grid cell (m)")
    ap.add_argument("--max-range", type=float, default=4.0, help="max depth counted as observed (m)")
    ap.add_argument("--depth-tol", type=float, default=0.05, help="occlusion tolerance (fraction of depth)")
    ap.add_argument("--min-gauss", type=int, default=30, help="min Gaussians for a cell to be scored")
    ap.add_argument("--weights", type=float, nargs=3, default=[0.35, 0.35, 0.30], help="w_views w_spread w_residual")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--min-dist", type=float, default=0.5, help="ignore Gaussians closer than this to any camera (floaters)")
    ap.add_argument("--cam-height", type=float, default=0.244, help="camera optical centre above the floor (m), from TF")
    ap.add_argument("--standoff", type=float, default=1.2, help="distance of the proposed viewpoint from the cell (m)")
    ap.add_argument("--max-detour", type=float, default=1.5, help="max distance of a viewpoint from the driven path (m)")
    ap.add_argument("--min-new-angle", type=float, default=30.0, help="min new viewing angle vs existing views (deg)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.result)
    calib = cfg["Dataset"]["Calibration"]
    ds = Path(cfg["Dataset"]["dataset_path"])
    pipe = munchify(cfg["pipeline_params"])
    g = GaussianModel(0, config=cfg)
    g.load_ply(str(args.result / "point_cloud" / "final" / "point_cloud.ply"))
    trj = json.load(open(args.result / "plot" / "trj_final.json"))
    cams = [np.array(p) for p in trj["trj_est"]]  # camera-to-world, world = odometry frame (first frame anchored)
    rgb_files = [l.split()[1] for l in open(ds / "rgb.txt") if l.strip() and not l.startswith("#")]
    images = [np.array(Image.open(ds / rgb_files[i]).convert("RGB")) for i in trj["trj_id"]]

    N = g.get_xyz.shape[0]
    n_views, res_sum, dirs = visibility_pass(g, cams, images, calib, pipe, args.max_range, args.depth_tol)
    spread = angular_spread(N, dirs)
    residual = np.where(n_views > 0, res_sum / np.maximum(n_views, 1), 0)
    xyz = g.get_xyz.detach().cpu().numpy()
    opa = g.get_opacity.detach().cpu().numpy().ravel()

    # floor grid in the odometry frame, Gaussians between floor and ~1.2 m above it
    cam_pos = np.array([c[:3, 3] for c in cams])
    z_cam = np.median(cam_pos[:, 2])
    floor_z = z_cam - args.cam_height  # odometry z of the floor
    band = (xyz[:, 2] > floor_z + 0.05) & (xyz[:, 2] < floor_z + 1.3) & (opa > 0.3) & (n_views > 0)  # objects, not floor
    # drop near-camera floaters: nothing real was within min_dist of the camera along the drive
    from scipy.spatial import cKDTree
    near, _ = cKDTree(cam_pos).query(xyz, k=1)
    band &= near > args.min_dist
    lo = np.floor(xyz[band, :2].min(0) / args.cell) * args.cell
    ij = np.floor((xyz[band, :2] - lo) / args.cell).astype(int)
    shape = ij.max(0) + 1
    w = opa[band]

    def pool(vals):
        num = np.zeros(shape)
        den = np.zeros(shape)
        np.add.at(num, (ij[:, 0], ij[:, 1]), vals * w)
        np.add.at(den, (ij[:, 0], ij[:, 1]), w)
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)

    count = np.zeros(shape, int)
    np.add.at(count, (ij[:, 0], ij[:, 1]), 1)
    V, A, Rr = pool(n_views[band]), pool(spread[band]), pool(residual[band])
    valid = count >= args.min_gauss
    Vn, An, Rn = (np.full(shape, np.nan) for _ in range(3))
    Vn[valid], An[valid], Rn[valid] = norm01(V[valid]), norm01(A[valid]), norm01(Rr[valid])
    wv, wa, wr = args.weights
    score = wv * (1 - Vn) + wa * (1 - An) + wr * Rn  # NaN where not enough evidence

    # occupancy (any Gaussian in band) to keep proposed viewpoints out of mapped geometry
    occ = count > 0

    def free(p):
        k = np.floor((p - lo) / args.cell).astype(int)
        if np.any(k < 0) or np.any(k >= shape):
            return True  # outside the mapped area: unknown, allowed (robot's own clear-ahead check applies)
        r = 1  # one-cell margin
        return not occ[max(k[0] - r, 0):k[0] + r + 1, max(k[1] - r, 0):k[1] + r + 1].any()

    # top cells (non-adjacent), each with a proposed viewpoint
    order = np.argsort(np.nan_to_num(score, nan=-1).ravel())[::-1]
    goals, taken = [], []
    for flat in order:
        if len(goals) >= args.top or np.isnan(score.ravel()[flat]):
            break
        ci = np.array(np.unravel_index(flat, shape))
        if any(np.abs(ci - t).max() <= 2 for t in taken):
            continue
        centre = lo + (ci + 0.5) * args.cell
        in_cell = band.copy()
        in_cell[band] = (ij[:, 0] == ci[0]) & (ij[:, 1] == ci[1])
        seen_bearings = [np.degrees(np.arctan2(*(-d[np.isin(i, np.nonzero(in_cell)[0])][:, [1, 0]].T)))
                         for i, d in dirs]
        seen = np.concatenate([b for b in seen_bearings if b.size]) if any(b.size for b in seen_bearings) else np.array([])
        best = None
        for th in np.radians(np.arange(0, 360, 10)):  # candidate viewpoints on a ring around the cell
            p = centre + args.standoff * np.array([np.cos(th), np.sin(th)])
            dist_path = np.min(np.linalg.norm(cam_pos[:, :2] - p, axis=1))
            if not free(p) or dist_path > args.max_detour:
                continue  # stay near observed, driven space; never plan into unseen territory
            bearing_from_cell = np.degrees(th)
            novelty = 180.0 if seen.size == 0 else np.min(np.abs((seen - bearing_from_cell + 180) % 360 - 180))
            if novelty < args.min_new_angle:
                continue  # too close to a view we already have
            # lateral revisits: reward new angle up to 90 deg, then nothing more; prefer short detours
            key = min(novelty, 90.0) - 30.0 * dist_path
            if best is None or key > best[0]:
                best = (key, p, novelty)
        if best is None:
            continue
        _, p, novelty = best
        yaw = np.arctan2(centre[1] - p[1], centre[0] - p[0])  # face the cell
        goals.append({"rank": len(goals) + 1, "cell_centre_xy": centre.round(3).tolist(), "score": float(score.ravel()[flat]),
                      "views": float(V[tuple(ci)]), "spread_deg": float(A[tuple(ci)]), "residual": float(Rr[tuple(ci)]),
                      "goal_xy": p.round(3).tolist(), "goal_yaw_rad": float(round(yaw, 4)),
                      "new_angle_deg": float(round(novelty, 1)),
                      "detour_m": float(round(np.min(np.linalg.norm(cam_pos[:, :2] - p, axis=1)), 2))})
        taken.append(ci)

    (args.out / f"{args.tag}_revisit.json").write_text(json.dumps(
        {"frame": "odometry (MonoGS world = first frame anchored to odom)", "cell_m": args.cell, "weights": args.weights,
         "n_gaussians": int(N), "n_keyframes": len(cams), "goals": goals}, indent=2))

    # figure: score heatmap (single-hue sequential), path, goals
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    seq = LinearSegmentedColormap.from_list("seq", ["#EEF2F8", "#9DB6D9", "#3F6FB0", "#1F2A4D"])
    ext = [lo[0], lo[0] + shape[0] * args.cell, lo[1], lo[1] + shape[1] * args.cell]
    span = np.array([ext[1] - ext[0], ext[3] - ext[2]])
    fig, ax = plt.subplots(figsize=(9.3, max(4.0, 9.3 * span[1] / span[0] + 1.2)), dpi=300)
    im = ax.imshow(score.T, origin="lower", extent=ext, cmap=seq, vmin=0, vmax=1, interpolation="nearest")
    ax.plot(cam_pos[:, 0], cam_pos[:, 1], "-", color="#D9822B", lw=3, label="driven path (keyframes)")
    for gl in goals:
        gx, gy = gl["goal_xy"]
        ax.annotate("", xy=gl["cell_centre_xy"], xytext=(gx, gy),
                    arrowprops=dict(arrowstyle="-|>", color="#1A1D24", lw=2, mutation_scale=18))
        ax.plot(gx, gy, "o", ms=13, mfc="white", mec="#1A1D24", mew=2)
        ax.text(gx, gy, str(gl["rank"]), ha="center", va="center", fontsize=11, weight="bold", color="#1A1D24")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=15)
    ax.set_ylabel("y (m)", fontsize=15)
    ax.tick_params(labelsize=13)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("revisit priority (0 = well observed, 1 = revisit)", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", fontsize=12, framealpha=0.95)
    ax.set_title(f"{args.tag}: revisit priority per {args.cell * 100:.0f} cm cell, top {len(goals)} suggested viewpoints",
                 fontsize=14, loc="left")
    fig.tight_layout()
    fig.savefig(args.out / f"{args.tag}_revisit.png", bbox_inches="tight")
    print(f"{args.tag}: {N} Gaussians, {len(cams)} keyframes, {int(valid.sum())} scored cells; goals:")
    for gl in goals:
        print(f"  #{gl['rank']} cell {gl['cell_centre_xy']} score {gl['score']:.2f} views {gl['views']:.1f} "
              f"spread {gl['spread_deg']:.0f} deg residual {gl['residual']:.3f} -> go to {gl['goal_xy']} "
              f"yaw {np.degrees(gl['goal_yaw_rad']):.0f} deg (new angle {gl['new_angle_deg']:.0f} deg, {gl['detour_m']} m off path)")


if __name__ == "__main__":
    main()
