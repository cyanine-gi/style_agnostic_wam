#!/usr/bin/env python
"""构建 Stage 1 隐空间缓存（E 冻结，一次性前向落盘）。

设计依据：guideline v2 §6-Stage 1 + 2026-09-15 裁决：
- 时间口径统一 15fps：real 名义 15fps（subsample=1），sim 30fps 抽帧到
  15fps（subsample=2）；先抽帧后运动抽稀（τ 语义在同速率流上对齐）；
- E 取 Stage 0 checkpoint 的权重（防漂移锚塑形后的表征），冻结前向；
- 每帧落盘三通道 latent + rgb + 64×64 disparity 目标 + 动作/本体感
  （详见 src/sawvla/data/latent_cache.py 模块头）。

用法：
    python scripts/build_latent_cache.py --stage0-ckpt outputs/stage0/last.pt
    python scripts/build_latent_cache.py --max-episodes 2 --max-frames-per-ep 40 \
        --outdir outputs/latent_cache_smoke     # 冒烟
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sawvla.data import Stage0MotionThinnedDataset, build_cache, write_index  # noqa: E402
from sawvla.models import DINOv2Encoder  # noqa: E402

# 2026-09-15 裁决：sim 30fps → 抽帧到名义 15fps 与 real 对齐
SUBSAMPLE = {"real": 1, "sim": 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-cfg", default="configs/model.yaml")
    ap.add_argument("--data-cfg", default="configs/data.yaml")
    ap.add_argument("--stage0-ckpt", type=Path,
                    default=Path("outputs/stage0/last.pt"))
    ap.add_argument("--outdir", type=Path, default=Path("outputs/latent_cache"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--motion-thresh", type=float, default=0.02,
                    help="运动抽稀阈值 τ（rad）；0=关闭抽稀（抽帧仍生效）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="每域最多处理多少 episode（冒烟用）")
    ap.add_argument("--max-frames-per-ep", type=int, default=None,
                    help="每 episode 最多缓存多少帧（冒烟用）")
    args = ap.parse_args()

    model_cfg = yaml.safe_load(open(args.model_cfg))
    data_cfg = yaml.safe_load(open(args.data_cfg))
    device = "cuda"

    E = DINOv2Encoder(model_cfg["paths"]["encoder"]).to(device)
    # 自己的 Stage 0 checkpoint（含 args 里的 PosixPath），可信源
    ckpt = torch.load(args.stage0_ckpt, map_location="cpu", weights_only=False)
    E.load_state_dict(ckpt["E"])
    E.freeze()
    E.eval()
    print(f"E <- {args.stage0_ckpt} (step {ckpt.get('step')})，冻结前向")

    t0 = time.time()
    records = []
    for tag in ("real", "sim"):
        dom = data_cfg["domains"][tag]
        ds = Stage0MotionThinnedDataset(
            root=dom["root"], domain_id=dom["domain_id"], camera=dom["camera"],
            depth_range=tuple(dom["depth_range"]), tasks=dom.get("tasks"),
            motion_thresh=args.motion_thresh, subsample=SUBSAMPLE[tag])
        records += build_cache(
            ds, E, args.outdir, tag, batch_size=args.batch, device=device,
            seed=args.seed, val_frac=args.val_frac,
            max_episodes=args.max_episodes,
            max_frames_per_ep=args.max_frames_per_ep)
        ds.close()

    write_index(args.outdir, records, meta={
        "stage0_ckpt": str(args.stage0_ckpt),
        "stage0_step": ckpt.get("step"),
        "motion_thresh": args.motion_thresh,
        "subsample": SUBSAMPLE,
        "fps_note": "real 名义 15fps；sim 30fps 抽帧到 15fps（2026-09-15 裁决）",
        "seed": args.seed, "val_frac": args.val_frac,
    })
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {args.outdir}")


if __name__ == "__main__":
    main()
