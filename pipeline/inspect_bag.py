"""Sanity-check a TurtleBot4 RGB-D bag (context doc section 6.10).

Checks: message counts and rates, RGB/depth timestamp pairing, depth validity and
centre-pixel distance, and that /tf + /tf_static connect odom -> base_link -> camera.
Saves a sample RGB frame and a colourised depth frame next to the bag.

Usage:  python tools/inspect_bag.py data/<bag_dir> [--frame N]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader

RGB = "/oakd_rgbd/rgb/image_raw"
DEPTH = "/oakd_rgbd/stereo/image_raw"
INFO = "/oakd_rgbd/rgb/camera_info"


def img_to_np(msg):
    if msg.encoding in ("bgr8", "rgb8"):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
        a = a.reshape(msg.height, msg.width, 3)
        return a[..., ::-1] if msg.encoding == "bgr8" else a
    if msg.encoding in ("16UC1", "mono16"):
        a = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)
        return a[:, : msg.width]
    raise ValueError(f"unsupported encoding {msg.encoding}")


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("--frame", type=int, default=None, help="frame index to save (default: middle)")
    args = ap.parse_args()

    rgb_t, depth_t, tf_edges = [], [], set()
    counts, first, last = {}, {}, {}
    rgb_keep, depth_keep, info = {}, {}, None

    with AnyReader([args.bag]) as reader:
        n_rgb = sum(c.msgcount for c in reader.connections if c.topic == RGB)
        want = args.frame if args.frame is not None else n_rgb // 2
        for conn, t, raw in reader.messages():
            topic = conn.topic
            counts[topic] = counts.get(topic, 0) + 1
            first.setdefault(topic, t)
            last[topic] = t
            if topic in (RGB, DEPTH, INFO, "/tf", "/tf_static"):
                msg = reader.deserialize(raw, conn.msgtype)
            else:
                continue
            if topic == RGB:
                if len(rgb_t) == want:
                    rgb_keep = {"msg": msg, "t": stamp(msg)}
                rgb_t.append(stamp(msg))
            elif topic == DEPTH:
                depth_t.append(stamp(msg))
                depth_keep[stamp(msg)] = msg if abs(len(depth_t) - 1 - want) <= 3 else None
            elif topic == INFO and info is None:
                info = msg
            elif topic in ("/tf", "/tf_static"):
                for tr in msg.transforms:
                    tf_edges.add((tr.header.frame_id, tr.child_frame_id))

    print(f"== {args.bag}")
    print("\n-- counts / rates (bag receive time)")
    for topic in sorted(counts):
        dur = (last[topic] - first[topic]) * 1e-9
        rate = (counts[topic] - 1) / dur if dur > 0 else 0
        print(f"{topic:40s} {counts[topic]:6d}  {rate:6.2f} Hz")

    rgb_t, depth_t = np.array(rgb_t), np.array(depth_t)
    print("\n-- RGB/depth pairing (header stamps)")
    if len(rgb_t) and len(depth_t):
        dt = np.array([np.min(np.abs(depth_t - t)) for t in rgb_t])
        print(f"rgb {len(rgb_t)}  depth {len(depth_t)}  ratio {len(depth_t) / len(rgb_t):.3f}")
        print(f"nearest-depth |dt|: median {np.median(dt) * 1e3:.1f} ms, 95% {np.percentile(dt, 95) * 1e3:.1f} ms, "
              f"max {dt.max() * 1e3:.1f} ms;  pairs within 30 ms: {np.mean(dt < 0.03) * 100:.1f}%")
    else:
        print("MISSING rgb or depth")

    if info is not None:
        k = np.array(info.k).reshape(3, 3)
        print(f"\n-- camera_info: {info.width}x{info.height} {info.distortion_model} "
              f"fx={k[0, 0]:.2f} fy={k[1, 1]:.2f} cx={k[0, 2]:.2f} cy={k[1, 2]:.2f}")
        print(f"   d = {np.round(np.array(info.d), 5).tolist()}")

    if rgb_keep:
        out = args.bag / "inspect"
        out.mkdir(exist_ok=True)
        rgb = img_to_np(rgb_keep["msg"])
        Image.fromarray(rgb).save(out / "rgb_sample.png")
        cands = {t: m for t, m in depth_keep.items() if m is not None}
        tdep = min(cands, key=lambda t: abs(t - rgb_keep["t"]))
        d = img_to_np(cands[tdep])
        valid = d > 0
        h, w = d.shape
        c = d[h // 2 - 5 : h // 2 + 5, w // 2 - 5 : w // 2 + 5]
        cv = c[c > 0]
        print(f"\n-- depth sample (frame {want}, |dt| {abs(tdep - rgb_keep['t']) * 1e3:.1f} ms)")
        print(f"valid pixels {valid.mean() * 100:.1f}%;  raw min/median/max of valid "
              f"{d[valid].min() if valid.any() else 0}/{int(np.median(d[valid])) if valid.any() else 0}/{d.max()}")
        print(f"centre 10x10 median raw = {int(np.median(cv)) if cv.size else 0}  "
              f"(if mm: {np.median(cv) / 1000 if cv.size else 0:.3f} m)  valid {cv.size}/100")
        dv = np.clip(d.astype(np.float32) / max(np.percentile(d[valid], 98), 1) if valid.any() else d * 0, 0, 1)
        import matplotlib.cm as cm
        col = (cm.turbo(dv)[..., :3] * 255).astype(np.uint8)
        col[~valid] = 0
        Image.fromarray(col).save(out / "depth_sample.png")
        Image.fromarray((0.5 * rgb + 0.5 * col).astype(np.uint8)).save(out / "overlay_sample.png")
        print(f"saved {out}/rgb_sample.png, depth_sample.png, overlay_sample.png")

    print("\n-- TF chain")
    children = {}
    for p, ch in tf_edges:
        children.setdefault(p, set()).add(ch)

    def path(a, b, seen=()):
        if a == b:
            return [a]
        for ch in children.get(a, ()):
            if ch not in seen:
                p = path(ch, b, seen + (a,))
                if p:
                    return [a] + p
        return None

    for a, b in [("odom", "base_link"), ("base_link", "oakd_link"),
                 ("oakd_link", "oakd_rgbd_rgb_camera_optical_frame"),
                 ("base_link", "oakd_rgb_camera_optical_frame")]:
        p = path(a, b)
        print(f"{a} -> {b}: {' -> '.join(p) if p else 'NOT CONNECTED'}")


if __name__ == "__main__":
    main()
