#!/usr/bin/env python
"""Stage 0 训练：E + D 深度塑形（隐空间还原深度图）快速验证。

设计依据：guideline v2 §5-Stage 0 / dataloader.md §8。
- 数据：两域经 Stage0MotionThinnedDataset（stage0_dataloader.py）按机械臂
  累计运动量贪心抽稀（默认 τ=0.02 rad，--motion-thresh 0 关闭），
  单帧（k=0），深度目标为**在线** 64×64 disparity（OnlineDisparitySupervision，
  已裁决 2026-09-11：快速验证不落盘；未来切离线产物只动 dataloader 装配）；
- 教师项关闭（λ_teacher=0）：教师离线产物尚未生成，即 depth_gt_supervision.md
  实现 #1 的 mask-only 基线配置；
- 防漂移锚（model.yaml encoder.anti_drift）：冻结 DINOv2 副本 +
  小权重特征蒸馏（按 encoder.py 约定在训练脚本侧构建）；
- train/val 按 episode 划分（dataloader.md §9）。快速验证期用
  crc32(episode_name) 确定性哈希划分（可复现、不落盘）；正式划分表产物
  待离线预处理落地时替换；
- bf16 autocast + 梯度累积；周期性 val + 深度可视化 + checkpoint；
- TensorBoard：标量（train/、val/）+ 深度对比图（val/viz）写入
  <outdir>/tb（jsonl 同时保留）；tensorboard 未安装时警告降级，不影响训练。

用法：
    python scripts/train_stage0.py --steps 200          # 快速验证
    python scripts/train_stage0.py --steps 100000 --batch 32 --accum 1
    tensorboard --logdir outputs/stage0/tb
"""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader, Subset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sawvla.data import Stage0MotionThinnedDataset  # noqa: E402
from sawvla.data.image import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from sawvla.losses import DepthLoss  # noqa: E402
from sawvla.models import DINOv2Encoder, DepthDecoder  # noqa: E402


def episode_split(ds, val_frac: float, seed: int):
    """按 episode 名 crc32 哈希确定性划分，返回 (train_idx, val_idx)。"""
    train, val = [], []
    offset = 0
    for ep_i, e in enumerate(ds.episodes):
        h = zlib.crc32(f"{seed}:{e['name']}".encode()) % 1000 / 1000
        (val if h < val_frac else train).extend(
            range(offset, offset + ds._frames_per_ep[ep_i]))
        offset += ds._frames_per_ep[ep_i]
    return train, val


def build_dataloaders(data_cfg, batch: int, workers: int, seed: int,
                      motion_thresh: float):
    """两域参数全部来自 configs/data.yaml；经 Stage0MotionThinnedDataset
    按运动量抽稀（motion_thresh<=0 时退化为全帧）。"""
    parts, train_idx, val_idx = [], [], []
    offset = 0
    for tag in ("real", "sim"):
        dom = data_cfg["domains"][tag]
        ds = Stage0MotionThinnedDataset(
            root=dom["root"], domain_id=dom["domain_id"], camera=dom["camera"],
            depth_range=tuple(dom["depth_range"]), tasks=dom.get("tasks"),
            motion_thresh=motion_thresh)
        tr, va = episode_split(ds, val_frac=0.1, seed=seed)
        train_idx += [i + offset for i in tr]
        val_idx += [i + offset for i in va]
        offset += len(ds)
        parts.append(ds)
    full = ConcatDataset(parts)
    train = DataLoader(Subset(full, train_idx), batch_size=batch,
                       shuffle=True, num_workers=workers,
                       pin_memory=True, drop_last=True)
    val = DataLoader(Subset(full, val_idx), batch_size=batch,
                     shuffle=False, num_workers=workers, pin_memory=True)
    return train, val, {t: len(d) for t, d in zip(("real", "sim"), parts)}


@torch.no_grad()
def validate(E, D, loss_fn, val_loader, device, max_batches: int = 20):
    E.eval()
    D.eval()
    tot, n = {}, 0
    for i, b in enumerate(val_loader):
        if i >= max_batches:
            break
        rgb = b["rgb"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = E(rgb)
            mu, log_sigma = D(z)
            parts = loss_fn(mu, log_sigma,
                            b["depth"].to(device), b["mask"].to(device),
                            torch.zeros_like(mu), rgb)
        for k, v in parts.items():
            tot[k] = tot.get(k, 0.0) + v.float().item()
        n += 1
    E.train()
    D.train()
    return {k: v / max(n, 1) for k, v in tot.items()}


@torch.no_grad()
def save_viz(E, D, val_loader, device, out: Path, n: int = 10,
             writer=None, step: int = 0):
    """从整个 val 集随机抽 n 帧可视化（两域按 val 集自然比例混合）。

    随机源以 step 为种子：不同 step 抽到不同帧（避免每次都看同一组），
    同一 step 可复现。"""
    E.eval()
    D.eval()
    ds = val_loader.dataset
    rng = np.random.default_rng(step)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    b = torch.utils.data.default_collate([ds[int(i)] for i in idx])
    n = len(idx)
    rgb = b["rgb"].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mu, _ = D(E(rgb))
    mu = mu.float().cpu().numpy()
    tgt = b["depth"].numpy()
    msk = b["mask"].numpy()
    # 每行一个样本：[原图 | target | 预测]，纵向排列（TB 滚动查看时
    # 一行即一组完整对比）
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.2 * n))
    for j in range(n):
        img = (rgb[j].cpu().numpy().transpose(1, 2, 0) * IMAGENET_STD
               + IMAGENET_MEAN).clip(0, 1)
        axes[j, 0].imshow(img)
        axes[j, 0].set_title(f"rgb dom={b['domain'][j]} ep{b['episode_id'][j]}",
                             fontsize=8)
        t = np.where(msk[j] > 0, tgt[j], np.nan)
        axes[j, 1].imshow(t, cmap="viridis")
        axes[j, 1].set_title("target disp (valid only)", fontsize=8)
        axes[j, 2].imshow(mu[j], cmap="viridis")
        axes[j, 2].set_title(f"pred mu [{mu[j].min():.2f},{mu[j].max():.2f}]",
                             fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    if writer is not None:
        writer.add_figure("val/viz", fig, step)
    plt.close(fig)
    E.train()
    D.train()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--motion-thresh", type=float, default=0.02,
                    help="运动抽稀阈值 τ（rad，14 关节 L2 累计）；0=关闭抽稀")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/stage0"))
    ap.add_argument("--model-cfg", default="configs/model.yaml")
    ap.add_argument("--data-cfg", default="configs/data.yaml")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    model_cfg = yaml.safe_load(open(args.model_cfg))
    data_cfg = yaml.safe_load(open(args.data_cfg))
    device = "cuda"

    train_loader, val_loader, sizes = build_dataloaders(
        data_cfg, args.batch, args.workers, args.seed, args.motion_thresh)
    print(f"data: {sizes}, train frames={len(train_loader.dataset)}, "
          f"val frames={len(val_loader.dataset)}")

    E = DINOv2Encoder(model_cfg["paths"]["encoder"]).to(device)
    # depth_decoder 节的键与 DepthDecoder.__init__ 参数一一对应；
    # d=384 / grid_size=16 用默认值（latent 节硬规格）
    D = DepthDecoder(**model_cfg["depth_decoder"]).to(device)
    # 防漂移锚：冻结副本（encoder.py 约定训练脚本侧构建）
    anchor = DINOv2Encoder(model_cfg["paths"]["encoder"]).to(device)
    anchor.freeze()
    anchor.eval()
    ad_cfg = model_cfg["encoder"]["anti_drift"]
    ad_w = ad_cfg["distill_weight"] if ad_cfg["enabled"] else 0.0

    loss_cfg = dict(model_cfg["loss"])
    n_grad_scales = loss_cfg.pop("n_grad_scales")
    loss_cfg["lambda_teacher"] = 0.0        # 教师产物未生成：mask-only 基线
    loss_fn = DepthLoss(n_grad_scales=n_grad_scales, **loss_cfg)

    opt = torch.optim.AdamW(
        [{"params": E.parameters(), "lr": model_cfg["encoder"]["lr"]},
         {"params": D.parameters(), "lr": args.lr}],
        weight_decay=0.01)

    log_path = args.outdir / "log.jsonl"
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.outdir / "tb")
        print(f"tensorboard: tensorboard --logdir {args.outdir / 'tb'}")
    except ImportError:
        print("警告：未安装 tensorboard，标量只写 log.jsonl")
        writer = None
    t0 = time.time()
    step = 0
    done = False
    while not done:
        for b in train_loader:
            rgb = b["rgb"].to(device, non_blocking=True)
            depth = b["depth"].to(device, non_blocking=True)
            mask = b["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = E(rgb)
                mu, log_sigma = D(z)
                parts = loss_fn(mu, log_sigma, depth, mask,
                                torch.zeros_like(mu), rgb)
                loss = parts["total"]
                if ad_w > 0:
                    with torch.no_grad():
                        z_ref = anchor(rgb)
                    parts["drift"] = torch.nn.functional.mse_loss(
                        z.float(), z_ref.float())
                    loss = loss + ad_w * parts["drift"]
            (loss / args.accum).backward()

            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(E.parameters()) + list(D.parameters()), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % 10 == 0:
                rec = {"step": step, "elapsed_s": round(time.time() - t0, 1),
                       **{k: round(v.float().item(), 5)
                          for k, v in parts.items()}}
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if writer is not None:
                    for k, v in parts.items():
                        writer.add_scalar(f"train/{k}", v.float().item(), step)
                print(rec)
            if step % args.val_every == 0 or step == args.steps - 1:
                va = validate(E, D, loss_fn, val_loader, device)
                if writer is not None:
                    for k, v in va.items():
                        writer.add_scalar(f"val/{k}", v, step)
                print(f"[val @ {step}] " + json.dumps(
                    {k: round(v, 5) for k, v in va.items()}))
                save_viz(E, D, val_loader, device,
                         args.outdir / f"viz_step{step:06d}.png",
                         writer=writer, step=step)
                torch.save({"E": E.state_dict(), "D": D.state_dict(),
                            "step": step, "args": vars(args)},
                           args.outdir / "last.pt")
            step += 1
            if step >= args.steps:
                done = True
                break

    if writer is not None:
        writer.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {args.outdir}")


if __name__ == "__main__":
    main()
