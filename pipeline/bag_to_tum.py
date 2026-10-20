"""Convert a TurtleBot4 RGB-D bag into a MonoGS TUM-format dataset + config.

Output (see notes/monogs_input_spec.md):
  <out>/rgb/<t>.png      8-bit RGB, undistorted
  <out>/depth/<t>.png    16-bit depth in mm (depth_scale 1000), undistorted with the same maps, 0 = invalid
  <out>/rgb.txt, depth.txt
  <out>/pose.txt         odom -> camera optical frame (camera-to-world), TUM format. MonoGS uses this ONLY
                         to place the first frame and for evaluation (ATE). It does not track with it.
  <out>/odom_base_link.txt  odom -> base_link, TUM format, for plots
  <cfg>                  MonoGS config (inherits configs/rgbd/tum/base_config.yaml)

Usage (in the monogs env):
  python bag_to_tum.py <bag_dir> [<bag_dir> ...] <out_dir> --config <cfg.yaml> [--start S] [--end S] [--stride N] [--max-depth M]
"""
import argparse
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation, Slerp

RGB = "/oakd_rgbd/rgb/image_raw"
DEPTH = "/oakd_rgbd/stereo/image_raw"
INFO = "/oakd_rgbd/rgb/camera_info"


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def tf_to_mat(tr):
    t, q = tr.transform.translation, tr.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def mat_to_tum(ts, T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x y z w
    return f"{ts:.6f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} {q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}"


def static_chain(static, parent, child):
    """Compose static transforms parent -> ... -> child (tree search)."""
    kids = {}
    for (p, c), T in static.items():
        kids.setdefault(p, []).append((c, T))

    def dfs(node, acc, seen):
        if node == child:
            return acc
        for c, T in kids.get(node, []):
            if c not in seen:
                r = dfs(c, acc @ T, seen | {c})
                if r is not None:
                    return r
        return None

    T = dfs(parent, np.eye(4), {parent})
    if T is None:
        raise RuntimeError(f"no static TF chain {parent} -> {child}")
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path, nargs="+", help="one or more bags, read as one sequence (robot must not be moved between them)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--config", type=Path, required=True, help="MonoGS config yaml to write")
    ap.add_argument("--dataset-path", default=None, help="dataset_path written into the config (default: out)")
    ap.add_argument("--start", type=float, default=0.0, help="seconds after first RGB frame")
    ap.add_argument("--end", type=float, default=1e9, help="seconds after first RGB frame")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-depth", type=float, default=8.0, help="metres; deeper/saturated depth -> 0")
    args = ap.parse_args()

    (args.out / "rgb").mkdir(parents=True, exist_ok=True)
    (args.out / "depth").mkdir(parents=True, exist_ok=True)

    info, static, odom = None, {}, []  # odom: (t, T odom->base_link)
    rgb_msgs, depth_index = [], {}
    recv_minus_hdr = {"rgb": [], "tf": []}

    with AnyReader(args.bag) as reader:
        conns = [c for c in reader.connections if c.topic in (RGB, DEPTH, INFO, "/tf", "/tf_static")]
        # pass 1: everything except image payloads
        t0 = None
        for conn, trecv, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == INFO and info is None:
                info = msg
            elif conn.topic == "/tf_static":
                for tr in msg.transforms:
                    static[(tr.header.frame_id, tr.child_frame_id)] = tf_to_mat(tr)
            elif conn.topic == "/tf":
                for tr in msg.transforms:
                    if tr.header.frame_id == "odom" and tr.child_frame_id == "base_link":
                        odom.append((stamp(tr.header), tf_to_mat(tr)))
                        recv_minus_hdr["tf"].append(trecv * 1e-9 - stamp(tr.header))
            elif conn.topic == RGB:
                ts = stamp(msg.header)
                t0 = ts if t0 is None else t0
                recv_minus_hdr["rgb"].append(trecv * 1e-9 - ts)
                if args.start <= ts - t0 <= args.end:
                    rgb_msgs.append((ts, msg))
            elif conn.topic == DEPTH:
                depth_index[stamp(msg.header)] = msg

    off = {k: np.median(v) for k, v in recv_minus_hdr.items() if v}
    print(f"receive-minus-header median: rgb {off.get('rgb', 0) * 1e3:.1f} ms, tf {off.get('tf', 0) * 1e3:.1f} ms")
    if abs(off.get("rgb", 0) - off.get("tf", 0)) > 0.05:
        print("WARNING: camera and odom clocks differ by >50 ms; pose association may be off")

    # camera intrinsics + undistortion (OAK-D: rational_polynomial, 8 coefs; MonoGS only does 5)
    W, H = info.width, info.height
    K = np.array(info.k, dtype=np.float64).reshape(3, 3)
    D = np.array(info.d, dtype=np.float64)
    Knew = K.copy()  # keep same intrinsics; just remove distortion
    m1, m2 = cv2.initUndistortRectifyMap(K, D, None, Knew, (W, H), cv2.CV_32FC1)
    disp = np.hypot(m1 - np.arange(W)[None, :], m2 - np.arange(H)[:, None])
    print(f"undistortion: max pixel shift {disp.max():.2f}, mean {disp.mean():.2f} ({info.distortion_model})")

    # extrinsic base_link -> camera optical frame (static)
    cam_frame = rgb_msgs[0][1].header.frame_id
    T_base_cam = static_chain(static, "base_link", cam_frame)
    print(f"base_link -> {cam_frame}: t = {np.round(T_base_cam[:3, 3], 3).tolist()}")

    odom.sort(key=lambda x: x[0])
    ot = np.array([o[0] for o in odom])
    rots = Rotation.from_matrix(np.stack([o[1][:3, :3] for o in odom]))
    slerp = Slerp(ot, rots)
    trans = np.stack([o[1][:3, 3] for o in odom])

    def odom_at(ts):
        if ts < ot[0] or ts > ot[-1]:
            return None
        T = np.eye(4)
        T[:3, :3] = slerp([ts]).as_matrix()[0]
        T[:3, 3] = [np.interp(ts, ot, trans[:, i]) for i in range(3)]
        return T

    dkeys = sorted(depth_index)
    rgb_lines, depth_lines, pose_lines, base_lines = [], [], [], []
    n_skip = 0
    for i, (ts, msg) in enumerate(rgb_msgs[:: args.stride]):
        j = bisect_left(dkeys, ts)
        cand = [k for k in (j - 1, j) if 0 <= k < len(dkeys)]
        td = min((dkeys[k] for k in cand), key=lambda k: abs(k - ts))
        T_ob = odom_at(ts)
        if abs(td - ts) > 0.03 or T_ob is None:
            n_skip += 1
            continue
        bgr = np.frombuffer(msg.data, np.uint8).reshape(H, msg.step)[:, : W * 3].reshape(H, W, 3)
        dm = depth_index[td]
        d = np.frombuffer(dm.data, np.uint16).reshape(dm.height, dm.step // 2)[:, : dm.width].copy()
        d[d > args.max_depth * 1000] = 0
        bgr_u = cv2.remap(bgr, m1, m2, cv2.INTER_LINEAR)
        d_u = cv2.remap(d, m1, m2, cv2.INTER_NEAREST)
        name = f"{ts:.6f}.png"
        cv2.imwrite(str(args.out / "rgb" / name), bgr_u)  # cv2 writes BGR -> file is RGB
        cv2.imwrite(str(args.out / "depth" / name), d_u)
        rgb_lines.append(f"{ts:.6f} rgb/{name}")
        depth_lines.append(f"{ts:.6f} depth/{name}")
        pose_lines.append(mat_to_tum(ts, T_ob @ T_base_cam))
        base_lines.append(mat_to_tum(ts, T_ob))

    (args.out / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    (args.out / "depth.txt").write_text("\n".join(depth_lines) + "\n")
    (args.out / "pose.txt").write_text("# timestamp tx ty tz qx qy qz qw  (odom -> camera optical, from wheel odometry)\n"
                                       + "\n".join(pose_lines) + "\n")
    (args.out / "odom_base_link.txt").write_text("# timestamp tx ty tz qx qy qz qw  (odom -> base_link)\n"
                                                 + "\n".join(base_lines) + "\n")

    P = np.array([[float(x) for x in l.split()[1:4]] for l in base_lines])
    path_len = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)) if len(P) > 1 else 0
    far = np.max(np.linalg.norm(P - P[0], axis=1)) if len(P) else 0
    print(f"frames written {len(rgb_lines)} (skipped {n_skip}); duration {float(rgb_lines[-1].split()[0]) - float(rgb_lines[0].split()[0]):.1f} s")
    print(f"odometry: path length {path_len:.2f} m, max distance from start {far:.2f} m, "
          f"end-to-start {np.linalg.norm(P[-1] - P[0]):.2f} m")

    ds = args.dataset_path or str(args.out)
    args.config.write_text(f"""inherit_from: "configs/rgbd/tum/base_config.yaml"

# Generated by turtlebot-gs/tools/bag_to_tum.py from {", ".join(b.name for b in args.bag)}
Dataset:
  dataset_path: "{ds}"
  Calibration:
    fx: {K[0, 0]:.6f}
    fy: {K[1, 1]:.6f}
    cx: {K[0, 2]:.6f}
    cy: {K[1, 2]:.6f}
    k1: 0.0
    k2: 0.0
    p1: 0.0
    p2: 0.0
    k3: 0.0
    distorted: False   # undistorted by the converter (OAK-D rational_polynomial)
    width: {W}
    height: {H}
    depth_scale: 1000.0   # OAK-D 16UC1 depth in mm; verified 2026-10-16 (1268 raw at ~1219 mm)
""")
    print(f"wrote {args.out} and {args.config}")


if __name__ == "__main__":
    main()
