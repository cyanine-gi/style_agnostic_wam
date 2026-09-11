#!/usr/bin/env python
"""两域全部相机视角对比图（dataloader.md §12.9 视角域差排查用）。

real（4 路）与 sim（6 路）各取一个 episode 的同一相对时刻帧，逐路渲染
并排对比，用于人工挑选两域视角最接近的相机配对。

用法：
    python scripts/check_camera_views.py [--ep 0] [--t 100]
输出：outputs/check_camera_views/view_compare.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sawvla.data.dataset import FrankaRealDataset, FrankaSimDataset, decode_color


def grab_all(ds, ep_i: int, t: int) -> dict[str, np.ndarray]:
    """取一个 episode 第 t 帧的全部相机 RGB（逐路按文件标记处理通道序）。"""
    e = ds.episodes[ep_i]
    out = {}
    with h5py.File(e["path"], "r") as f:
        imgs = f["camera_observations/color_images"]
        for cam in sorted(imgs.keys()):
            n = imgs[cam].shape[0]
            chan_key = f"camera_color_channel/{cam}"
            chan = f[chan_key][()].decode() if chan_key in f else "bgr"
            out[cam] = decode_color(imgs[cam][min(t, n - 1)], chan)
    return out


def labeled(img: np.ndarray, text: str, cell: tuple[int, int]) -> np.ndarray:
    """统一缩放到 cell (w, h)（保持比例、填黑）并加标题条。"""
    cw, chh = cell
    bar = 36
    s = min(cw / img.shape[1], (chh - bar) / img.shape[0])
    im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    canvas = np.zeros((chh, cw, 3), np.uint8)
    canvas[bar:bar + im.shape[0], :im.shape[1]] = im
    cv2.putText(canvas, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--t", type=int, default=100)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/check_camera_views/view_compare.png"))
    args = ap.parse_args()

    real = FrankaRealDataset()
    sim = FrankaSimDataset()
    domains = [("real", real, grab_all(real, args.ep, args.t)),
               ("sim", sim, grab_all(sim, args.ep, args.t))]

    cell = (640, 400)
    rows = []
    for name, ds, imgs in domains:
        tiles = [labeled(im, f"{name}/{cam}  {im.shape[1]}x{im.shape[0]}", cell)
                 for cam, im in imgs.items()]
        rows.append(np.hstack(tiles))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    grid = np.vstack(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"-> {args.out}  (real ep={real.episodes[args.ep]['name']}, "
          f"sim ep={sim.episodes[args.ep]['name']}, t={args.t})")
    real.close()
    sim.close()


if __name__ == "__main__":
    main()
