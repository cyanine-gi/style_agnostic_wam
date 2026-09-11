#!/usr/bin/env python
"""Visualize RoboMIND2.0 HDF5 trajectories (e.g. tienkung_sim / tienkung).

Usage:
    # interactive player (camera streams + joint curves with time cursor)
    python scripts/visualize_robomind_hdf5.py \
        data/RoboMIND2.0/data/tienkung_sim/7109000-2025_09_01_13_12_28.hdf5

    # export an mp4 montage instead of opening a window
    python scripts/visualize_robomind_hdf5.py <file.hdf5> --save out.mp4

    # only dump a structural summary
    python scripts/visualize_robomind_hdf5.py <file.hdf5> --summary-only

Keyboard (interactive mode): space = play/pause, left/right = step 1 frame,
up/down = step 10 frames, q = quit.
"""

import argparse
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


# --------------------------------------------------------------------------- #
# data access helpers
# --------------------------------------------------------------------------- #
def decode_color(buf: np.ndarray) -> np.ndarray:
    """JPEG byte buffer -> RGB image."""
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_depth(buf: np.ndarray) -> np.ndarray:
    """PNG byte buffer -> depth image (uint16 or uint8)."""
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)


def list_cameras(f: h5py.File):
    return sorted(f["camera_observations/color_images"].keys())


def get_time(f: h5py.File) -> np.ndarray:
    """Relative timestamps in seconds (auto-detects ms vs s vs us epoch)."""
    ts = f["camera_observations/timestamp"][:].astype(np.float64)
    t = ts - ts[0]
    dt = np.median(np.diff(t))
    if dt > 1e4:      # micro/nanoseconds or ms-scale epoch deltas
        t /= 1e3
    elif dt > 1.0:    # milliseconds
        t /= 1e3
    return t


def load_state_curves(f: h5py.File) -> dict:
    """Collect every master/puppet state group that has a non-empty 2-D `data`."""
    curves = {}
    for side in ("master", "puppet"):
        if side not in f:
            continue
        for name in f[side]:
            g = f[f"{side}/{name}"]
            if "data" not in g:
                continue
            d = g["data"]
            if d.ndim == 2 and d.shape[0] > 0:
                curves[f"{side}/{name}"] = d[:]
    return curves


def print_summary(path: str, f: h5py.File, cams, curves, t):
    print(f"\n{'=' * 72}\n{path}\n{'=' * 72}")
    if "metadata" in f:
        for k in f["metadata"]:
            v = f[f"metadata/{k}"][()]
            if isinstance(v, bytes):
                v = v.decode("utf-8", "replace")
            print(f"  metadata/{k}: {v}")
    print(f"  frames: {len(t)}   duration: {t[-1]:.2f}s   "
          f"fps: {(len(t) - 1) / max(t[-1], 1e-9):.1f}")
    print(f"  cameras: {cams}")
    for cam in cams:
        res = f[f"camera_color_resolution/{cam}"][:]
        print(f"    {cam}: color resolution {tuple(res)}")
    print("  state curves:")
    for k, v in curves.items():
        print(f"    {k}: {v.shape} {v.dtype}")


# --------------------------------------------------------------------------- #
# interactive player
# --------------------------------------------------------------------------- #
def play(path: str, fps: float):
    with h5py.File(path, "r") as f:
        cams = list_cameras(f)
        t = get_time(f)
        curves = load_state_curves(f)
        print_summary(path, f, cams, curves, t)

        n = len(t)
        color_ds = {c: f[f"camera_observations/color_images/{c}"] for c in cams}
        depth_ds = {c: f[f"camera_observations/depth_images/{c}"] for c in cams}

        instr = ""
        if "metadata" in f and "language_instruction" in f["metadata"]:
            instr = f["metadata/language_instruction"][()]
            instr = instr.decode("utf-8", "replace") if isinstance(instr, bytes) else str(instr)

        # ---- layout: color row, depth row, then one row of state curves ---- #
        n_cols = max(len(cams), 1)
        curve_keys = sorted(curves.keys())
        n_curve_rows = len(curve_keys)
        fig = plt.figure(figsize=(20 * n_cols, 12 * (2 + n_curve_rows)))
        gs = GridSpec(2 + n_curve_rows, n_cols, figure=fig, hspace=0.35, wspace=0.15)
        fig.suptitle(f"{Path(path).name}   |   \"{instr}\"", fontsize=11)

        img_axes, img_artists, depth_axes, depth_artists = [], [], [], []
        for i, cam in enumerate(cams):
            ax = fig.add_subplot(gs[0, i])
            ax.set_title(cam)
            ax.axis("off")
            img_artists.append(ax.imshow(decode_color(color_ds[cam][0])))
            img_axes.append(ax)

            axd = fig.add_subplot(gs[1, i])
            axd.set_title(f"{cam} depth")
            axd.axis("off")
            depth_artists.append(
                axd.imshow(decode_depth(depth_ds[cam][0]), cmap="turbo"))
            depth_axes.append(axd)

        cursors = []
        for r, key in enumerate(curve_keys):
            ax = fig.add_subplot(gs[2 + r, :])
            data = curves[key]
            tc = np.linspace(t[0], t[-1], data.shape[0])
            ax.plot(tc, data, lw=0.8)
            ax.set_xlim(t[0], t[-1])
            ax.set_ylabel(key.replace("/", "\n"), fontsize=7, rotation=0,
                          ha="right", va="center")
            ax.grid(alpha=0.3)
            cursors.append(ax.axvline(t[0], color="k", ls="--", lw=1))

        state = {"idx": 0, "playing": True}

        time_txt = fig.text(0.01, 0.01, "", fontsize=9)

        def render(idx):
            for cam, art, dart in zip(cams, img_artists, depth_artists):
                art.set_data(decode_color(color_ds[cam][idx]))
                d = decode_depth(depth_ds[cam][idx])
                dart.set_data(d)
                dart.set_clim(d.min(), max(d.max(), d.min() + 1))
            for cur in cursors:
                cur.set_xdata([t[idx], t[idx]])
            time_txt.set_text(
                f"frame {idx + 1}/{n}   t = {t[idx]:.2f}s   "
                f"{'playing' if state['playing'] else 'paused'}")
            fig.canvas.draw_idle()

        def on_key(ev):
            if ev.key == " ":
                state["playing"] = not state["playing"]
            elif ev.key == "right":
                state["idx"] = min(state["idx"] + 1, n - 1)
            elif ev.key == "left":
                state["idx"] = max(state["idx"] - 1, 0)
            elif ev.key == "up":
                state["idx"] = min(state["idx"] + 10, n - 1)
            elif ev.key == "down":
                state["idx"] = max(state["idx"] - 10, 0)
            elif ev.key == "q":
                plt.close(fig)
                return
            if ev.key != " ":
                state["playing"] = False
            render(state["idx"])

        fig.canvas.mpl_connect("key_press_event", on_key)

        timer = fig.canvas.new_timer(interval=int(1000 / fps))
        def tick():
            if state["playing"] and state["idx"] < n - 1:
                state["idx"] += 1
                render(state["idx"])
        timer.add_callback(tick)
        timer.start()

        # ---- enlarge: scale figure so top color images render 1:1 on screen ----
        fig.canvas.draw()
        bbox = img_axes[0].get_window_extent()  # current on-screen size in px
        ih, iw = decode_color(color_ds[cams[0]][0]).shape[:2]
        w_in, h_in = fig.get_size_inches()
        fig.set_size_inches(w_in * iw / bbox.width, h_in * ih / bbox.height)

        render(0)
        plt.show()


# --------------------------------------------------------------------------- #
# mp4 export
# --------------------------------------------------------------------------- #
def save_video(path: str, out: str, fps: float):
    with h5py.File(path, "r") as f:
        cams = list_cameras(f)
        t = get_time(f)
        print_summary(path, f, cams, load_state_curves(f), t)
        n = len(t)
        color_ds = {c: f[f"camera_observations/color_images/{c}"] for c in cams}
        depth_ds = {c: f[f"camera_observations/depth_images/{c}"] for c in cams}

        writer = None
        for i in range(n):
            rows = []
            for decode, dss in ((decode_color, color_ds), (None, depth_ds)):
                tiles = []
                for cam in cams:
                    if decode is decode_color:
                        img = cv2.cvtColor(decode(dss[cam][i]), cv2.COLOR_RGB2BGR)
                    else:
                        d = decode_depth(dss[cam][i]).astype(np.float32)
                        d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
                        img = cv2.applyColorMap((d * 255).astype(np.uint8),
                                                cv2.COLORMAP_TURBO)
                    cv2.putText(img, f"{cam} {'depth' if decode is None else 'color'}",
                                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)
                    tiles.append(img)
                h = max(im.shape[0] for im in tiles)
                tiles = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
                         for im in tiles]
                rows.append(np.hstack(tiles))
            w = max(r.shape[1] for r in rows)
            rows = [cv2.resize(r, (w, int(r.shape[0] * w / r.shape[1])))
                    for r in rows]
            frame = np.vstack(rows)

            if writer is None:
                writer = cv2.VideoWriter(
                    out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (frame.shape[1], frame.shape[0]))
            cv2.putText(frame, f"frame {i + 1}/{n}  t={t[i]:.2f}s",
                        (8, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            writer.write(frame)
            if (i + 1) % 100 == 0:
                print(f"  wrote {i + 1}/{n} frames")
        writer.release()
        print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hdf5", help="path to a RoboMIND2.0 .hdf5 trajectory")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="playback / export frame rate (default 30)")
    ap.add_argument("--save", metavar="OUT.mp4",
                    help="export a camera montage video instead of opening the player")
    ap.add_argument("--summary-only", action="store_true",
                    help="print the file summary and exit")
    args = ap.parse_args()

    if args.summary_only:
        with h5py.File(args.hdf5, "r") as f:
            print_summary(args.hdf5, f, list_cameras(f), load_state_curves(f),
                          get_time(f))
    elif args.save:
        save_video(args.hdf5, args.save, args.fps)
    else:
        play(args.hdf5, args.fps)


if __name__ == "__main__":
    main()
