#!/usr/bin/env python
"""可视化 put_egg_into_box（real）与 sim 对照任务的前几帧。

背景（2026-09-11）：real put_egg_into_box 的 EE 字段为空（300/300 episode），
是 matched 任务中 EE 缺失的一侧。本脚本把该任务头几帧的 RGB（经方案 D
预处理，即模型实际所见）与 EE/关节信号并排画出，供人工判断：
- 缺失 EE 的 real episode 里夹爪动作是否可见、是否关键；
- real/sim 对应任务的初始场景与手形态差异。

产出：``<outdir>/put_egg_first_frames.png``
行 = episode（上 real 下 sim），列 = 帧；每格下方标注帧号与 EE 值。

用法：
    python scripts/visualize_put_egg.py --episodes 3 --frames 6 --stride 15
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sawvla.data import RoboMindDataset  # noqa: E402
from sawvla.data.image import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

REAL_TASK = "put_egg_into_box"
SIM_TASK = "110-pack_egg_into_box_and_close_lid"


def ee_value(ds: RoboMindDataset, ep_idx: int, t: int):
    """读 puppet EE 标量；空字段返回 None。"""
    import h5py
    e = ds.episodes[ep_idx]
    with h5py.File(e["path"], "r") as f:
        vals = []
        for side in ("left", "right"):
            key = f"puppet/end_effector_{side}_position_align/data"
            if key in f and f[key].size > 0:
                a = f[key][:].reshape(-1, f[key].shape[-1] if f[key].ndim > 1 else 1)
                vals.append(f"{a[t].mean():.2f}")
            else:
                vals.append("n/a")
        return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--stride", type=int, default=15,
                    help="帧间隔（默认 15，覆盖开头 ~1.5–3 秒）")
    ap.add_argument("--offset", type=int, default=0,
                    help="起始帧偏移（默认 0=开头；渲染抓取窗口用）")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/vis_put_egg"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    domains = [
        ("real", RoboMindDataset("data/RoboMIND2.0-Tienkung", domain_id=0,
                                 camera="camera_top", tasks=[REAL_TASK])),
        ("sim", RoboMindDataset("data/RoboMIND2.0-Tienkung-sim", domain_id=1,
                                camera="camera_head", tasks=[SIM_TASK])),
    ]

    n_rows = sum(min(args.episodes, len(ds.episodes)) for _, ds in domains)
    fig, axes = plt.subplots(n_rows, args.frames,
                             figsize=(3.2 * args.frames, 3.0 * n_rows))
    axes = np.atleast_2d(axes)

    row = 0
    for tag, ds in domains:
        for ep in range(min(args.episodes, len(ds.episodes))):
            e = ds.episodes[ep]
            n = ds._frames_per_ep[ep]
            ts = np.minimum(args.offset + np.arange(args.frames) * args.stride,
                            n - 1)
            for col, t in enumerate(ts):
                s = ds[int(ds._cum[ep]) + int(t)]
                rgb = (s["rgb"].numpy().transpose(1, 2, 0) * IMAGENET_STD
                       + IMAGENET_MEAN).clip(0, 1)
                ee_l, ee_r = ee_value(ds, ep, int(t))
                ax = axes[row, col]
                ax.imshow(rgb)
                ax.set_title(f"f{t}  EE[L,R]=[{ee_l},{ee_r}]", fontsize=8)
                ax.axis("off")
            axes[row, 0].set_ylabel(f"{tag}\n{e['name']}", fontsize=8,
                                    rotation=0, ha="right", va="center")
            row += 1
        ds.close()

    fig.tight_layout()
    out = args.outdir / f"put_egg_frames_o{args.offset}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
