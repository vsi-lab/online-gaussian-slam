"""Poster metrics table + pipeline diagram (PNG, 300 dpi). Reads numbers from results/*/ json files.

Usage: python tools/poster_tables.py   (from turtlebot-gs/)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "poster" / "figures"


def row(name, data, d, n_frames, note):
    r = json.load(open(ROOT / "results" / d / "render_after_opt.json"))
    s = json.load(open(ROOT / "results" / d / "stats_final.json"))
    return [name, data, f"{s['rmse'] * 100:.1f}", f"{r['mean_psnr']:.1f}", f"{r['mean_ssim']:.2f}",
            f"{r['mean_lpips']:.2f}", str(n_frames), note]


rows = [
    row("TUM fr1_desk", "TUM benchmark", "tum_fr1_desk", 573, "vs. mocap ground truth"),
    row("run_01", "TurtleBot4 (ours)", "run_01_clip", 300, "vs. wheel odometry"),
    row("run_02", "TurtleBot4 (ours)", "run_02_clip", 292, "vs. wheel odometry"),
]
cols = ["Sequence", "Data", "ATE RMSE (cm)", "PSNR (dB)", "SSIM", "LPIPS", "Frames", "Trajectory reference"]

fig, ax = plt.subplots(figsize=(10, 1.6), dpi=300)
ax.axis("off")
t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(9)
t.scale(1, 1.6)
t.auto_set_column_width(list(range(len(cols))))
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    if r == 0:
        cell.set_facecolor("#1f3b5c")
        cell.set_text_props(color="white", weight="bold")
    elif r == 1:
        cell.set_facecolor("#f1f1f1")
fig.text(0.5, 0.02, "MonoGS RGB-D, RTX 5080. Real-robot rows: 1.8 m straight drive at 0.15 m/s, OAK-D Pro 640x360 @ 15 Hz. "
         "Rendering metrics after final optimisation on training views. Odometry is not ground truth.",
         ha="center", fontsize=6.5, color="#444444")
fig.savefig(OUT / "metrics_table.png", bbox_inches="tight")
plt.close(fig)

# ---- pipeline diagram
steps = [
    ("TurtleBot4", "OAK-D Pro RGB-D\nCreate 3 odometry"),
    ("ros2 bag on robot", "RGB + aligned depth 15 Hz\n/tf, /imu, /scan"),
    ("bag_to_tum.py", "undistort, pair by stamp,\nmm depth (scale 1000)"),
    ("MonoGS (RGB-D)", "Gaussian Splatting SLAM\ntracks its own poses"),
    ("Gaussian map", "renders, novel views,\nATE vs odometry"),
]
fig, ax = plt.subplots(figsize=(11, 1.9), dpi=300)
ax.set_xlim(0, len(steps) * 2.2)
ax.set_ylim(0, 2)
ax.axis("off")
for i, (title, sub) in enumerate(steps):
    x = i * 2.2 + 0.1
    ax.add_patch(FancyBboxPatch((x, 0.35), 1.8, 1.3, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc="#1f3b5c" if i in (0, 4) else "#e8eef5", ec="#1f3b5c", lw=1.2))
    col = "white" if i in (0, 4) else "#1f3b5c"
    ax.text(x + 0.9, 1.3, title, ha="center", va="center", fontsize=9, weight="bold", color=col)
    ax.text(x + 0.9, 0.78, sub, ha="center", va="center", fontsize=7, color=col)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + 1.92, 1.0), (x + 2.28, 1.0), arrowstyle="-|>", mutation_scale=14,
                                     color="#1f3b5c", lw=1.4))
ax.text(len(steps) * 1.1, 0.08, "Odometry is used only to place the first frame and as an evaluation yardstick; "
        "MonoGS estimates every later pose itself.", ha="center", fontsize=7, color="#444444", style="italic")
fig.savefig(OUT / "pipeline_diagram.png", bbox_inches="tight")
plt.close(fig)
print("saved metrics_table.png, pipeline_diagram.png")
