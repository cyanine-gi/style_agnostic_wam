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
  小权重特征蒸馏（覆盖 patch + 全部 register 槽位，2026-09-14 裁决）；
- 全局信号注册表（model.yaml signals / src/sawvla/signals）：register
  分区 [0,1]=域 / [2,3]=域无关，信号头只读域无关槽位（当前 joint_pos）；
- 域探针（DomainProbe，独立组件）：patch / reg_domain / reg_agnostic
  三通道域分类准确率只监控不反传（TB: probe/acc_*）——先量后治；
- train/val 按 episode 划分（dataloader.md §9）。快速验证期用
  crc32(episode_name) 确定性哈希划分（可复现、不落盘）；正式划分表产物
  待离线预处理落地时替换；
- bf16 autocast + 梯度累积；周期性 val + 深度可视化 + checkpoint；
- checkpoint（2026-09-15）：`--save-every`（默认 1000）步存
  `ckpt_step{step:06d}.pt` 全量检查点（模型 + 主/探针优化器状态 + RNG），
  `last.pt` 同格式随 val 更新；`--resume <ckpt>` 一键恢复继续训练
  （注意：恢复后数据洗牌顺序从新 epoch 开始，不逐 batch 复现原顺序）；
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
from sawvla.models import DINOv2Encoder, DepthDecoder, DomainProbe  # noqa: E402
from sawvla.signals import SignalContext, build_signals  # noqa: E402


def full_ckpt(step: int, models: dict, opts: dict, args) -> dict:
    """全量检查点：模型权重 + 优化器状态（含 AdamW 动量/lr）+ RNG。"""
    return {**{k: m.state_dict() for k, m in models.items()},
            **{f"opt/{k}": o.state_dict() for k, o in opts.items()},
            "rng": {"cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state()},
            "step": step, "args": vars(args)}


def load_full_ckpt(path, models: dict, opts: dict, device) -> int:
    """加载全量检查点，返回继续训练的起始 step（= 保存 step + 1）。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)  # 自己的 ckpt
    for k, m in models.items():
        m.load_state_dict(ck[k])
    for k, o in opts.items():
        o.load_state_dict(ck[f"opt/{k}"])
        for st in o.state.values():          # 优化器状态搬回 GPU
            for kk, v in st.items():
                if torch.is_tensor(v):
                    st[kk] = v.to(device)
    if "rng" in ck:
        torch.set_rng_state(ck["rng"]["cpu"])
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        np.random.set_state(ck["rng"]["numpy"])
    return ck["step"] + 1


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
def validate(E, D, loss_fn, val_loader, device, heads=None, roles=None,
             max_batches: int = 20):
    E.eval()
    D.eval()
    tot, n = {}, 0
    for i, b in enumerate(val_loader):
        if i >= max_batches:
            break
        rgb = b["rgb"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok = E.forward_tokens(rgb)
            mu, log_sigma = D(tok["patch"])
            parts = loss_fn(mu, log_sigma,
                            b["depth"].to(device), b["mask"].to(device),
                            torch.zeros_like(mu), rgb)
            if heads is not None and len(heads.heads) > 0:
                ctx = SignalContext(**tok, roles=roles)
                for k, v in heads.losses(ctx, b).items():
                    parts[f"signal/{k}"] = v
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


@torch.no_grad()
def probe_heatmap(probe, E, val_loader, device, max_batches: int = 10):
    """val 集上累计逐 token 域分类准确率 → (grid, grid) 泄漏热图。"""
    E.eval()
    tot, n = None, 0
    for i, b in enumerate(val_loader):
        if i >= max_batches:
            break
        rgb = b["rgb"].to(device, non_blocking=True)
        domain = b["domain"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok = E.forward_tokens(rgb)
        acc = probe.token_accuracies(tok["patch"], domain)
        tot = acc if tot is None else tot + acc
        n += 1
    E.train()
    m = (tot / max(n, 1)).cpu().numpy()
    return m.reshape(probe.grid_size, probe.grid_size)


def save_probe_map(m, out: Path, writer=None, step: int = 0):
    """渲染泄漏热图：红=域可分（泄漏），白=≈50% 随机（干净）。"""
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(m, cmap="Reds", vmin=0.5, vmax=1.0)
    ax.set_title(f"patch domain leak map mean={m.mean():.3f} max={m.max():.3f}",
                 fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    if writer is not None:
        writer.add_figure("probe/patch_leak_map", fig, step)
    plt.close(fig)


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
    ap.add_argument("--save-every", type=int, default=1000,
                    help="每多少步存 ckpt_step{step}.pt 全量检查点")
    ap.add_argument("--resume", type=Path, default=None,
                    help="全量检查点路径：加载模型+优化器+RNG 继续训练")
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

    # 全局信号注册表（register 分区：roles 决定信号头读哪几个槽位）
    roles = model_cfg["encoder"]["register_roles"]
    heads = build_signals(model_cfg, dim=model_cfg["latent"]["dim"],
                          n_reg=E.num_registers).to(device)
    print(f"signals: {[h.name for h in heads.heads]} "
          f"(register roles: {roles})")
    # 域探针（独立仪器）：三通道域分类准确率，只监控不反传进 E；
    # 外加 patch 逐 token 泄漏热图（grid_size 启用，2026-09-15 裁决）
    probe = DomainProbe(dim=model_cfg["latent"]["dim"],
                        grid_size=model_cfg["latent"]["grid_size"]).to(device)
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    loss_cfg = dict(model_cfg["loss"])
    n_grad_scales = loss_cfg.pop("n_grad_scales")
    loss_cfg["lambda_teacher"] = 0.0        # 教师产物未生成：mask-only 基线
    loss_fn = DepthLoss(n_grad_scales=n_grad_scales, **loss_cfg)

    opt = torch.optim.AdamW(
        [{"params": E.parameters(), "lr": model_cfg["encoder"]["lr"]},
         {"params": D.parameters(), "lr": args.lr},
         {"params": heads.parameters(), "lr": args.lr}],
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
    start_step = 0
    if args.resume:
        start_step = load_full_ckpt(
            args.resume,
            {"E": E, "D": D, "signals": heads, "probe": probe},
            {"main": opt, "probe": probe_opt}, device)
        print(f"resume <- {args.resume}，从 step {start_step} 继续"
              f"（数据从新 epoch 洗牌，不逐 batch 复现原顺序）")
        if start_step >= args.steps:
            raise SystemExit(
                f"resume step {start_step} >= --steps {args.steps}，无需训练")
    step = start_step
    done = False
    while not done:
        for b in train_loader:
            rgb = b["rgb"].to(device, non_blocking=True)
            depth = b["depth"].to(device, non_blocking=True)
            mask = b["mask"].to(device, non_blocking=True)
            domain = b["domain"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = E.forward_tokens(rgb)
                ctx = SignalContext(**tok, roles=roles)
                mu, log_sigma = D(tok["patch"])
                parts = loss_fn(mu, log_sigma, depth, mask,
                                torch.zeros_like(mu), rgb)
                loss = parts["total"]
                # 注册的全局信号（读域无关 register），加权进总损失
                for k, v in heads.losses(ctx, b).items():
                    parts[f"signal/{k}"] = v
                    loss = loss + v
                if ad_w > 0:
                    with torch.no_grad():
                        tok_ref = anchor.forward_tokens(rgb)
                    # 锚覆盖 patch + 全部 register（2026-09-14 裁决）；
                    # CLS 无人消费，不锚
                    z_all = torch.cat([tok["patch"], tok["registers"]], dim=1)
                    z_ref = torch.cat([tok_ref["patch"], tok_ref["registers"]],
                                      dim=1)
                    parts["drift"] = torch.nn.functional.mse_loss(
                        z_all.float(), z_ref.float())
                    loss = loss + ad_w * parts["drift"]
            (loss / args.accum).backward()

            # 域探针：独立小优化器，特征在探针内部 detach，梯度不进 E
            probe_losses = probe.losses(ctx, domain)
            sum(probe_losses.values()).backward()
            probe_opt.step()
            probe_opt.zero_grad(set_to_none=True)

            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(E.parameters()) + list(D.parameters())
                    + list(heads.parameters()), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % 10 == 0:
                accs = probe.accuracies(ctx, domain)
                tok_acc = probe.token_accuracies(tok["patch"], domain)
                rec = {"step": step, "elapsed_s": round(time.time() - t0, 1),
                       **{k: round(v.float().item(), 5)
                          for k, v in parts.items()},
                       **{f"probe/acc_{k}": round(v, 4)
                          for k, v in accs.items()},
                       "probe/tok_acc_mean": round(tok_acc.mean().item(), 4),
                       "probe/tok_acc_max": round(tok_acc.max().item(), 4)}
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if writer is not None:
                    for k, v in parts.items():
                        writer.add_scalar(f"train/{k}", v.float().item(), step)
                    for k, v in accs.items():
                        writer.add_scalar(f"probe/acc_{k}", v, step)
                    writer.add_scalar("probe/tok_acc_mean",
                                      tok_acc.mean().item(), step)
                    writer.add_scalar("probe/tok_acc_max",
                                      tok_acc.max().item(), step)
                print(rec)
            if step % args.val_every == 0 or step == args.steps - 1:
                va = validate(E, D, loss_fn, val_loader, device,
                              heads=heads, roles=roles)
                if writer is not None:
                    for k, v in va.items():
                        writer.add_scalar(f"val/{k}", v, step)
                print(f"[val @ {step}] " + json.dumps(
                    {k: round(v, 5) for k, v in va.items()}))
                save_viz(E, D, val_loader, device,
                         args.outdir / f"viz_step{step:06d}.png",
                         writer=writer, step=step)
                hm = probe_heatmap(probe, E, val_loader, device)
                save_probe_map(hm, args.outdir / f"probe_map_step{step:06d}.png",
                               writer=writer, step=step)
                torch.save(full_ckpt(step,
                                     {"E": E, "D": D, "signals": heads,
                                      "probe": probe},
                                     {"main": opt, "probe": probe_opt}, args),
                           args.outdir / "last.pt")
            if step > 0 and step % args.save_every == 0:
                torch.save(full_ckpt(step,
                                     {"E": E, "D": D, "signals": heads,
                                      "probe": probe},
                                     {"main": opt, "probe": probe_opt}, args),
                           args.outdir / f"ckpt_step{step:06d}.pt")
            step += 1
            if step >= args.steps:
                done = True
                break

    if writer is not None:
        writer.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {args.outdir}")


if __name__ == "__main__":
    main()
