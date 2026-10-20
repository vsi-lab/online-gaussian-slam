"""Find bright, well-reconstructed keyframes and try wider overview renders (run in ~/MonoGS, PYTHONPATH=.).

Usage:
  python hero_search.py scan --result <stamp_dir> [<stamp_dir> ...]
  python hero_search.py make --result <stamp_dir> --kf K --out <png> [--back 0.6] [--fov-scale 0.7] [--gamma 0.8]
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


def load(result):
    cfg = load_cfg(result)
    calib = dict(cfg["Dataset"]["Calibration"])
    g = GaussianModel(0, config=cfg)
    g.load_ply(str(result / "point_cloud" / "final" / "point_cloud.ply"))
    trj = json.load(open(result / "plot" / "trj_final.json"))
    ds = Path(cfg["Dataset"]["dataset_path"])
    rgb = [l.split()[1] for l in open(ds / "rgb.txt") if l.strip() and not l.startswith("#")]
    return cfg, calib, munchify(cfg["pipeline_params"]), g, [np.array(p) for p in trj["trj_est"]], trj["trj_id"], ds, rgb


def psnr(a, b):
    return 10 * np.log10(255.0**2 / max(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2), 1e-6))


def scan(args):
    bg = torch.zeros(3, device="cuda")
    for r in args.result:
        cfg, calib, pipe, g, est, ids, ds, rgb = load(r)
        rows = []
        for k, (c2w, i) in enumerate(zip(est, ids)):
            real = np.array(Image.open(ds / rgb[i]).convert("RGB"))
            ren = render_rgb(g, make_cam(c2w, calib), pipe, bg)
            rows.append((k, i, real.mean(), psnr(real, ren), (real.std())))
        rows.sort(key=lambda x: -(x[2] / 255 * 40 + x[3]))
        print(f"== {r.name} ({ds.name})  kf, frame, brightness, PSNR, contrast")
        for row in rows[:8]:
            print("  kf %2d frame %4d  bright %5.1f  psnr %5.1f  std %5.1f" % row)


def label(img, text):
    img = np.ascontiguousarray(img)
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, col, th, cv2.LINE_AA)
    return img


def make(args):
    bg = torch.zeros(3, device="cuda")
    cfg, calib, pipe, g, est, ids, ds, rgb = load(args.result[0])
    c2w = est[args.kf]
    real = np.array(Image.open(ds / rgb[ids[args.kf]]).convert("RGB"))
    ren = render_rgb(g, make_cam(c2w, calib), pipe, bg)
    p = psnr(real, ren)
    # overview: pulled back along the view direction with a wider field of view (same height: raised views fail)
    wide = dict(calib)
    wide["fx"], wide["fy"] = calib["fx"] * args.fov_scale, calib["fy"] * args.fov_scale
    if args.kf2 is not None:  # a different part of the room: render at another keyframe's pose
        over = render_rgb(g, make_cam(est[args.kf2], calib), pipe, bg)
        real2 = np.array(Image.open(ds / rgb[ids[args.kf2]]).convert("RGB"))
        p2 = psnr(real2, over)
    else:
        over = render_rgb(g, make_cam(offset_pose(c2w, back=args.back), wide), pipe, bg)
    lut = (np.linspace(0, 1, 256) ** args.gamma * 255).astype(np.uint8)  # same display curve on every panel
    real_b, ren_b, over_b = lut[real], lut[ren], lut[over]
    strip = np.concatenate([label(real_b, "robot camera frame"),
                            label(ren_b, f"online Gaussian map, same pose ({p:.1f} dB)"),
                            label(over_b, f"same map, other side of the room ({p2:.1f} dB)" if args.kf2 is not None
                                  else "map overview: pulled back, wider view")], 1)
    Image.fromarray(strip).save(args.out, dpi=(300, 300))
    print(f"saved {args.out}: kf {args.kf}, frame PSNR {p:.2f} dB (computed before display gamma {args.gamma})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "make"])
    ap.add_argument("--result", type=Path, nargs="+", required=True)
    ap.add_argument("--kf", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--back", type=float, default=0.6)
    ap.add_argument("--fov-scale", type=float, default=0.7)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--kf2", type=int, default=None, help="third panel: render at this keyframe instead of an overview")
    args = ap.parse_args()
    scan(args) if args.mode == "scan" else make(args)


if __name__ == "__main__":
    main()
