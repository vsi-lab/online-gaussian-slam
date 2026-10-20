"""MonoGS vs COLMAP+3DGS vs GLOMAP+3DGS on our TurtleBot runs: table + time chart + trajectory overlay.

Run in WSL (monogs env) from ~/MonoGS:
  python compare_figures.py --out /mnt/c/Users/uoavs/turtlebot-gs/poster/figures --res /mnt/c/Users/uoavs/turtlebot-gs/results/compare
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_poses import colmap_centres, ref_positions, umeyama  # noqa: E402

HOME = Path.home()
RUNS = {
    "run_01": dict(ds="datasets/turtlebot/run_01_clip", sfm=HOME / "sfm_compare/run_01_clip",
                   monogs="results/datasets_turtlebot/2026-10-16-11-54-28", log=HOME / "monogs_run_01_clip.log",
                   label="Straight 1.8 m (300 frames)"),
    "run_05_L": dict(ds="datasets/turtlebot/run_05_L", sfm=HOME / "sfm_compare/run_05_L",
                     monogs="results/datasets_turtlebot/2026-10-16-14-39-47", log=HOME / "monogs_run05L.log",
                     label="L-route 5.2 m with 90° turn (1303 frames)"),
}
# validated categorical palette (validate_palette.js, light): all checks pass; orange contrast WARN -> direct labels + table
COL = {"MonoGS": "#1f6fd1", "COLMAP": "#d9822b", "GLOMAP": "#2a9d74"}
MARK = {"MonoGS": "o", "COLMAP": "s", "GLOMAP": "^"}
INK, MUTED, GRID = "#1A1D24", "#4A4F5C", "#E3E4E8"
DPI = 300
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def timing(p):
    return {l.split()[0]: float(l.split()[1]) for l in open(p / "timing.txt") if len(l.split()) == 2}


def gs_metrics(p, m):
    t = open(p / f"metrics_3dgs_{m}.txt").read()
    g = lambda k: float(re.search(rf"{k}\s*:\s*([0-9.]+)", t).group(1))
    return {"psnr": g("PSNR"), "ssim": g("SSIM"), "lpips": g("LPIPS")}


def monogs_time(log):
    t = open(log).read().replace("\r", "\n")
    return float(re.findall(r"Total time ([0-9.]+)", t)[-1])


def collect(res_dir):
    rows = {}
    for r, c in RUNS.items():
        t = timing(c["sfm"])
        poses = json.load(open(res_dir / f"poses_{r}.json"))
        mg = json.load(open(Path(c["monogs"]) / "psnr" / "after_opt" / "final_result.json"))
        n = int(t["images"])
        front = t["features_s"] + t["matching_s"]
        rows[r] = {
            "frames": n,
            "MonoGS": dict(online=True, pose_s=monogs_time(c["log"]), train_s=0.0, front_s=0.0,
                           reg=n, ate_sim3=poses["monogs"]["sim3"]["rmse_m"], ate_metric=poses["monogs"]["se3"]["rmse_m"],
                           psnr=mg["mean_psnr"], ssim=mg["mean_ssim"], lpips=mg["mean_lpips"]),
        }
        for m, key in (("COLMAP", "colmap"), ("GLOMAP", "glomap")):
            g = gs_metrics(c["sfm"], key)
            rows[r][m] = dict(online=False, front_s=front, pose_s=t[f"{key}_mapper_s"], train_s=t[f"3dgs_train_{key}_s"],
                              reg=int(t[f"{key}_registered"]), ate_sim3=poses[key]["sim3"]["rmse_m"], ate_metric=None, **g)
    return rows


def fmt_s(s):
    return f"{s / 60:.1f} min" if s >= 120 else f"{s:.0f} s"


S = {"table_fs": 8.5, "table_size": (13, 3.2), "foot": 6.3, "time_size": (10, 3.4), "tick": 9, "lab": 8.5,
     "title": 9.5, "val": 8.5, "traj_size": (6.2, 5.2), "leg": 7.5, "ms": 5, "sfx": ""}
POSTER = {"table_fs": 21, "table_size": (30, 7.2), "foot": 15, "time_size": (9.3, 6.2), "tick": 17, "lab": 16,
          "title": 18, "val": 16, "traj_size": (9.3, 8.4), "leg": 14, "ms": 9, "sfx": "_poster"}


def table_fig(rows, out):
    cols = ["Route", "Method", "Online?", "Time to map", "Registered", "ATE, scale-fitted", "ATE, metric",
            "PSNR (dB)", "SSIM", "LPIPS"]
    cells, colours = [], []
    for r, c in RUNS.items():
        for m in ("MonoGS", "COLMAP", "GLOMAP"):
            d = rows[r][m]
            total = d["front_s"] + d["pose_s"] + d["train_s"]
            cells.append([c["label"] if m == "MonoGS" else "", m, "yes" if d["online"] else "no (batch)", fmt_s(total),
                          f"{d['reg']}/{rows[r]['frames']}", f"{d['ate_sim3'] * 100:.1f} cm",
                          f"{d['ate_metric'] * 100:.1f} cm" if d["ate_metric"] is not None else "n/a (no scale)",
                          f"{d['psnr']:.1f}", f"{d['ssim']:.2f}", f"{d['lpips']:.2f}"])
            colours.append(COL[m])
    fig, ax = plt.subplots(figsize=S["table_size"], dpi=DPI)
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(S["table_fs"])
    t.scale(1, 1.55)
    t.auto_set_column_width(list(range(len(cols))))
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor("#C9CBD1")
        if i == 0:
            cell.set_facecolor("#1F2A4D")
            cell.set_text_props(color="white", weight="bold")
        elif i in (4, 5, 6):
            cell.set_facecolor("#F4F5F7")
        if i > 0 and j == 1:
            cell.set_text_props(weight="bold", color=INK)
    fig.text(0.5, 0.01,
             "Real TurtleBot4 data, RTX 5080. Time to map: MonoGS = online SLAM wall time; COLMAP/GLOMAP = SIFT + matching + mapper "
             "+ 3DGS training (30k iters). ATE vs wheel odometry (not ground truth); SfM is scale-free, so only scale-fitted ATE applies.\n"
             "Image metrics: 3DGS on every 8th registered frame held out; MonoGS on non-keyframes at its own poses. "
             "GLOMAP = COLMAP 4.2 global_mapper. COLMAP's L-route metrics cover only its 670 registered frames.",
             ha="center", va="bottom", fontsize=S["foot"], color=MUTED, wrap=True)
    fig.savefig(out / f"compare_table{S['sfx']}.png", bbox_inches="tight")
    plt.close(fig)


def time_fig(rows, out):
    fig, axes = plt.subplots(*((2, 1) if S["sfx"] else (1, 2)), figsize=S["time_size"], dpi=DPI)
    for ax, (r, c) in zip(axes, RUNS.items()):
        ms = ["MonoGS", "COLMAP", "GLOMAP"]
        y = np.arange(len(ms))[::-1]
        for yi, m in zip(y, ms):
            d = rows[r][m]
            left = 0
            parts = [("features + matching", d["front_s"], 0.35), ("poses", d["pose_s"], 1.0), ("3DGS training", d["train_s"], 0.6)]
            if m == "MonoGS":
                parts = [("online SLAM (poses + map)", d["pose_s"], 1.0)]
            for name, v, a in parts:
                if v <= 0:
                    continue
                ax.barh(yi, v / 60, left=left / 60, height=0.55, color=COL[m], alpha=a, edgecolor="white", linewidth=2)
                left += v
            ax.text(left / 60 + 0.4, yi, fmt_s(left), va="center", fontsize=S["val"], color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(ms, fontsize=S["tick"], color=INK)
        ax.tick_params(axis="x", labelsize=S["tick"] - 2)
        ax.set_xlabel("minutes to a finished Gaussian map", fontsize=S["lab"])
        ax.set_title(c["label"], fontsize=S["title"], color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(rows[r][m]["front_s"] + rows[r][m]["pose_s"] + rows[r][m]["train_s"] for m in ms) / 60 * 1.25)
    if not S["sfx"]:
        fig.text(0.5, -0.04, "Bars for COLMAP / GLOMAP: light = SIFT + matching, solid = pose estimation, mid = 3DGS training. "
                 "MonoGS produces poses and map while frames arrive; COLMAP/GLOMAP need the whole recording first.",
                 ha="center", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / f"compare_time{S['sfx']}.png", bbox_inches="tight")
    plt.close(fig)


def traj_fig(out):
    c = RUNS["run_05_L"]
    ds = Path(c["ds"])
    ref = ref_positions(ds)
    keys_all = sorted(ref)
    odom = np.array([ref[k] for k in keys_all])
    fig, ax = plt.subplots(figsize=S["traj_size"], dpi=DPI)
    ax.plot(odom[:, 0], odom[:, 1], "--", color="#2C3140", lw=1.4, label="wheel odometry (reference)", zorder=4)
    for m, key in (("COLMAP", "colmap"), ("GLOMAP", "glomap")):
        cc = colmap_centres(c["sfm"] / key / "txt")
        ks = sorted(k for k in cc if k in ref)
        est, gt = np.array([cc[k] for k in ks]), np.array([ref[k] for k in ks])
        s, R, t = umeyama(est, gt, True)
        al = (s * (R @ est.T)).T + t
        ax.plot(al[:, 0], al[:, 1], MARK[m], color=COL[m], ms=1.8 * (1.7 if S["sfx"] else 1), alpha=0.75, zorder=2,
                label=f"{m} ({len(ks)}/{len(keys_all)} frames, scale-fitted)")
    trj = json.load(open(Path(c["monogs"]) / "plot" / "trj_final.json"))
    est = np.array([np.array(p)[:3, 3] for p in trj["trj_est"]])
    gt = np.array([np.array(p)[:3, 3] for p in trj["trj_gt"]])
    s, R, t = umeyama(est, gt, False)
    al = (R @ est.T).T + t
    ax.plot(al[:, 0], al[:, 1], "-" + MARK["MonoGS"], color=COL["MonoGS"], ms=S["ms"], lw=1.6 * (1.6 if S["sfx"] else 1), zorder=3,
            mec="white", mew=1.0, label="MonoGS keyframes (metric, no scale fit)")
    ax.annotate("start", odom[0, :2], xytext=(-6, 14), textcoords="offset points", fontsize=S["lab"], color=INK)
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(odom[:, :2], axis=0), axis=1))]
    corner = odom[np.argmin(np.abs(dist - 2.74)), :2]  # end of leg 1 (2.74 m)
    ax.annotate("90° turn", corner, xytext=(10, -18), textcoords="offset points", fontsize=S["lab"], color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("end", odom[-1, :2], xytext=(8, 0), textcoords="offset points", fontsize=S["lab"], color=INK)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=S["lab"])
    ax.set_ylabel("y (m)", fontsize=S["lab"])
    ax.tick_params(labelsize=S["tick"] - 2)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("L-route: estimated camera paths vs wheel odometry", fontsize=S["title"], color=INK, loc="left")
    ax.legend(fontsize=S["leg"], loc="upper left", frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / f"compare_traj_L{S['sfx']}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--poster", action="store_true", help="print-size figures with large type (*_poster.png)")
    args = ap.parse_args()
    if args.poster:
        S.update(POSTER)
    rows = collect(args.res)
    (args.res / "compare_summary.json").write_text(json.dumps(rows, indent=2))
    table_fig(rows, args.out)
    time_fig(rows, args.out)
    traj_fig(args.out)
    for r in rows:
        for m in ("MonoGS", "COLMAP", "GLOMAP"):
            d = rows[r][m]
            print(r, m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})


if __name__ == "__main__":
    main()
