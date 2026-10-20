"""Novel-view degradation sweep: render the map from viewpoints increasingly far from the robot's path.

Rows = offset direction (up / right), columns = offset distance. Each tile is rendered from a mid-path
keyframe pose moved by that offset (no rotation), so tile 0 m is a view the robot actually had.
Shows how quickly reconstruction quality falls off away from observed viewpoints, which motivates
choosing revisit views (Prof. Lee's question).

Usage (in ~/MonoGS, monogs env, PYTHONPATH=.):
  python offset_sweep.py --result results/datasets_turtlebot/<stamp> --tag run_01 --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from munch import munchify
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from make_figures import load_cfg, make_cam, offset_pose, render_rgb  # noqa: E402

from gaussian_splatting.scene.gaussian_model import GaussianModel  # noqa: E402

DIST = [0.0, 0.1, 0.2, 0.35]
DIRS = {"up": dict(up=1), "right": dict(right=1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.result)
    calib = cfg["Dataset"]["Calibration"]
    pipe = munchify(cfg["pipeline_params"])
    gauss = GaussianModel(0, config=cfg)
    gauss.load_ply(str(args.result / "point_cloud" / "final" / "point_cloud.ply"))
    bg = torch.zeros(3, device="cuda")
    est = [np.array(p) for p in json.load(open(args.result / "plot" / "trj_final.json"))["trj_est"]]
    base = est[len(est) // 3]

    rows = []
    for name, unit in DIRS.items():
        tiles = []
        for d in DIST:
            img = render_rgb(gauss, make_cam(offset_pose(base, **{k: v * d for k, v in unit.items()}), calib), pipe, bg)
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
            txt = "on path" if d == 0 else f"{d:.2f} m {name}"
            cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        rows.append(np.concatenate(tiles, axis=1))
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(args.out / f"{args.tag}_offset_sweep.png", dpi=(300, 300))
    print(f"saved {args.out / f'{args.tag}_offset_sweep.png'}")


if __name__ == "__main__":
    main()
