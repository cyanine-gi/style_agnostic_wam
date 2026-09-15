#!/usr/bin/env python
"""Stage 1 训练：动作条件转移模型 T（z_t + a_t → ẑ_{t+1}），register 递推。

设计依据：guideline v2 §6-Stage 1 + 2026-09-15 用户裁决：
- **上下文 2 帧**（滑窗；1 帧对物体速度不可观测）；clip 不跨 episode；
- **自由展开**（feed-back，非 teacher forcing）：上下文块带真值 z/r/proprio，
  展开块回喂 ẑ/r̂、只带动作 token（未来本体感不可知 ⇒ 可学习 null 向量）；
  固定最大展开 K=4 步、逐步监督（等效覆盖裁决的 unroll k∈{1,2,4}）；
- **register 递推**：T 同时预测 ẑ（patch）与 r̂（register），两通道都计 L_dyn；
- **预测端监督对等（选项 A）**：joint_pos 等注册信号同样挂 r̂ 且**梯度进 T**
  （grad_into_t: false 即未来的选项 C 消融）；深度 D 同样解码 ẑ 但
  **对 T stop-grad**（只训 D + 监控，DINO-WM 消融惯例 §2.2）；
- E 冻结不进训练循环——输入来自 latent 缓存
  （scripts/build_latent_cache.py，real 15fps / sim 抽帧到 15fps）；
- 域探针挂在**预测输出**上（E 冻结 ⇒ z 的探针读数是常数，无意义；
  ẑ 的泄漏量才是 Stage 1 的监控对象）；
- checkpoint（2026-09-16）：`--save-every`（默认 1000）步存
  `ckpt_step{step:06d}.pt` 全量检查点（模型 + 主/探针优化器状态 + RNG），
  `last.pt` 同格式随 val 更新；`--resume <ckpt>` 一键恢复继续训练
  （注意：恢复后数据洗牌顺序从新 epoch 开始，不逐 batch 复现原顺序）。

用法：
    python scripts/train_stage1.py --steps 200 --cache-dir outputs/latent_cache_smoke
    python scripts/train_stage1.py --steps 150000 --batch 8
    python scripts/train_stage1.py --overfit-batch   # 单 batch 过拟合 sanity
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sawvla.data import LatentClipDataset  # noqa: E402
from sawvla.losses import DepthLoss, latent_prediction_loss  # noqa: E402
from sawvla.models import (ActionAdapter, DepthDecoder, DomainProbe,  # noqa: E402
                           TransitionModel)
from sawvla.signals import SignalContext, build_signals  # noqa: E402


# --------------------------------------------------------------------------- #
# 全量检查点（模型 + 优化器 + RNG；--resume 一键继续训练）
# --------------------------------------------------------------------------- #

def full_ckpt(step: int, models: dict, opts: dict, args) -> dict:
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


# --------------------------------------------------------------------------- #
# 单 batch 前向与损失（train / val 共用）
# --------------------------------------------------------------------------- #

def rollout(T, adapter, b, C: int, K: int, device):
    """自由展开：2 帧真值上下文 → 逐步回喂 ẑ/r̂，返回 K 步预测。

    返回 preds：长度 K 的列表，第 i 项 = (ẑ, r̂) 对第 C+i 帧的预测。
    """
    L = C + K
    z, r = b["patch"].to(device).float(), b["registers"].to(device).float()
    actions = b["action"].to(device).float()
    proprio = b["proprio"].to(device).float()
    # block j 携带 a_j / proprio_j（产生第 j+1 帧），共需 L-1 个 block
    act_tok, prop_tok = adapter(actions[:, :L - 1], proprio[:, :L - 1])
    B = z.shape[0]

    z_in, r_in = z[:, :C], r[:, :C]
    pv = torch.ones(B, C, dtype=torch.bool, device=device)
    preds = []
    for j in range(C, L):
        out = T.forward_with_registers(
            z_in, r_in, act_tok[:, :j], prop_tok[:, :j], pv, frame_offset=0)
        preds.append((out["patch"][:, -1], out["registers"][:, -1]))
        if j < L - 1:                            # 回喂，准备下一展开块
            z_in = torch.cat([z_in, out["patch"][:, -1:]], dim=1)
            r_in = torch.cat([r_in, out["registers"][:, -1:]], dim=1)
            pv = torch.cat([pv, torch.zeros(B, 1, dtype=torch.bool,
                                            device=device)], dim=1)
    return preds


def compute_losses(preds, b, C: int, D, heads, roles,
                   loss_fn, s1_cfg, device, probe=None):
    """对 K 步预测计算全部损失；返回 (总损失, 日志用 parts)。

    选项 A 梯度口径（2026-09-15 裁决）：
    - L_dyn（patch + register）：目标一律 sg（缓存本身无梯度），塑形 T；
    - 信号损失：真值侧只训头（E 冻结、缓存无梯度），预测侧梯度进 T
      （grad_into_t: false 时预测侧 detach——选项 C 消融）；
    - 深度损失：D 解码真值 z 与预测 ẑ（一律 detach），只训练 D。
    """
    K = len(preds)
    depth_pred_on = s1_cfg["depth_on_prediction"]["enabled"]
    sig_pred_on = s1_cfg["signals_on_prediction"]["enabled"]
    sig_grad_t = s1_cfg["signals_on_prediction"]["grad_into_t"]
    reg_w = float(s1_cfg["reg_dyn_weight"])

    z = b["patch"].to(device).float()
    r = b["registers"].to(device).float()
    cls = b["cls"].to(device).float()
    proprio = b["proprio"].to(device).float()
    domain = b["domain"].to(device)

    parts: dict[str, torch.Tensor] = {}
    loss = 0.0
    probe_losses = []

    for i, (z_hat, r_hat) in enumerate(preds):
        t = C + i                                # 目标帧索引
        # --- L_dyn：patch + register（动态区域加权用真值相邻帧差） ---
        parts[f"dyn/patch_{i}"] = latent_prediction_loss(
            z_hat, z[:, t], z_prev=z[:, t - 1])
        parts[f"dyn/reg_{i}"] = latent_prediction_loss(
            r_hat, r[:, t], z_prev=r[:, t - 1])
        loss = loss + parts[f"dyn/patch_{i}"] + reg_w * parts[f"dyn/reg_{i}"]

        ctx_hat = SignalContext(
            cls=torch.zeros_like(cls[:, t]),     # T 不预测 CLS（无人消费）
            registers=r_hat if sig_grad_t else r_hat.detach(),
            patch=z_hat if sig_grad_t else z_hat.detach(),
            roles=roles)
        ctx_true = SignalContext(cls=cls[:, t], registers=r[:, t],
                                 patch=z[:, t], roles=roles)
        tgt_batch = {"proprio": proprio[:, t]}

        # --- 信号：真值侧训头；预测侧（选项 A）塑形 T ---
        if len(heads.heads) > 0:
            for k_, v in heads.losses(ctx_true, tgt_batch).items():
                parts[f"signal_true/{k_}_{i}"] = v
                loss = loss + v
            if sig_pred_on:
                for k_, v in heads.losses(ctx_hat, tgt_batch).items():
                    parts[f"signal_pred/{k_}_{i}"] = v
                    loss = loss + v

        # --- 深度：D 解码 ẑ（detach，只训 D）+ 真值 z 保持 D 校准 ---
        if depth_pred_on:
            rgb_t = b["rgb"][:, t].to(device).float() / 255.0
            d_t = b["depth"][:, t].to(device)
            m_t = b["mask"][:, t].to(device).float()
            mu_p, ls_p = D(z_hat.detach())
            p_pred = loss_fn(mu_p, ls_p, d_t, m_t,
                             torch.zeros_like(mu_p), rgb_t)
            mu_t, ls_t = D(z[:, t])
            p_true = loss_fn(mu_t, ls_t, d_t, m_t,
                             torch.zeros_like(mu_t), rgb_t)
            parts[f"depth/pred_{i}"] = p_pred["total"]
            parts[f"depth/true_{i}"] = p_true["total"]
            loss = loss + 0.5 * (p_pred["total"] + p_true["total"])

        # --- 探针：挂预测输出（E 冻结，z 侧读数是常数） ---
        if probe is not None:
            probe_losses.append(probe.losses(ctx_hat, domain))

    # 汇总（日志精简：逐步明细进 parts，总量取均值）
    summary = {
        "dyn/patch": sum(parts[f"dyn/patch_{i}"] for i in range(K)) / K,
        "dyn/reg": sum(parts[f"dyn/reg_{i}"] for i in range(K)) / K,
    }
    for prefix in ("signal_true", "signal_pred", "depth/pred", "depth/true"):
        keys = [k for k in parts if k.startswith(prefix)]
        if keys:
            summary[prefix] = sum(parts[k] for k in keys) / len(keys)
    return loss, summary, probe_losses


# --------------------------------------------------------------------------- #
# 验证与可视化
# --------------------------------------------------------------------------- #

@torch.no_grad()
def validate(T, adapter, D, heads, roles, loss_fn, probe, val_loader,
             C, K, device, s1_cfg, max_batches: int = 10):
    T.eval()
    D.eval()
    tot, step_err, step_err_r, n = {}, None, None, 0
    probe_acc, norm_ratio = None, 0.0
    for i, b in enumerate(val_loader):
        if i >= max_batches:
            break
        preds = rollout(T, adapter, b, C, K, device)
        _, summary, _ = compute_losses(
            preds, b, C, D, heads, roles, loss_fn, s1_cfg, device)
        for k, v in summary.items():
            tot[k] = tot.get(k, 0.0) + v.float().item()
        z = b["patch"].to(device).float()
        r = b["registers"].to(device).float()
        domain = b["domain"].to(device)
        se = [((ph - z[:, C + j]) ** 2).mean().item()
              for j, (ph, _) in enumerate(preds)]
        se_r = [((rh - r[:, C + j]) ** 2).mean().item()
                for j, (_, rh) in enumerate(preds)]
        step_err = se if step_err is None else [a + b_ for a, b_ in
                                                zip(step_err, se)]
        step_err_r = se_r if step_err_r is None else [a + b_ for a, b_ in
                                                      zip(step_err_r, se_r)]
        norm_ratio += (preds[-1][0].norm(dim=-1).mean()
                       / z[:, C + K - 1].norm(dim=-1).mean().clamp(min=1e-6)
                       ).item()
        # 探针读数（预测输出，逐步平均）
        ctx_last = SignalContext(cls=torch.zeros_like(r[:, 0, 0]),
                                 registers=preds[-1][1], patch=preds[-1][0],
                                 roles=roles)
        pa = probe.accuracies(ctx_last, domain)
        probe_acc = pa if probe_acc is None else {
            k: probe_acc[k] + pa[k] for k in pa}
        n += 1
    T.train()
    D.train()
    out = {k: v / max(n, 1) for k, v in tot.items()}
    for j in range(K):
        out[f"latent_mse/step_{j + 1}"] = step_err[j] / max(n, 1)
        out[f"latent_mse_reg/step_{j + 1}"] = step_err_r[j] / max(n, 1)
    out["z_norm_ratio_last"] = norm_ratio / max(n, 1)
    for k in (probe_acc or {}):
        out[f"probe/acc_{k}"] = probe_acc[k] / max(n, 1)
    return out


@torch.no_grad()
def save_viz(T, adapter, D, val_ds, C, K, device, out_prefix: Path,
             writer=None, step: int = 0, n_clips: int = 100,
             per_page: int = 25, batch: int = 32):
    """随机抽 n_clips 个起点帧（以 step 为种子：同 step 可复现、不同 step
    不同帧，与 Stage 0 约定一致），每个起点一行：
    [起点 rgb | +1..+K target（集中） | +1..+K pred（集中）]——
    真值/预测各成一组，同一步数在镜像列位上，便于对比运动走向。
    分页输出（每页 per_page 行），PNG + TB。"""
    T.eval()
    D.eval()
    rng = np.random.default_rng(step)
    idx = rng.choice(len(val_ds), size=min(n_clips, len(val_ds)),
                     replace=False)
    items = [val_ds[int(i)] for i in idx]
    # 分批 rollout，收集各步 D(ẑ)
    mu_pred = [[] for _ in range(K)]
    for b0 in range(0, len(items), batch):
        b = torch.utils.data.default_collate(items[b0: b0 + batch])
        preds = rollout(T, adapter, b, C, K, device)
        for i, (z_hat, _) in enumerate(preds):
            mu, _ = D(z_hat.detach())
            mu_pred[i].append(mu.float().cpu().numpy())
    mu_pred = [np.concatenate(m) for m in mu_pred]       # 每步 (n, 64, 64)

    n = len(items)
    rgb0 = np.stack([it["rgb"][C - 1].numpy().transpose(1, 2, 0) / 255.0
                     for it in items])                   # 起点帧 = 最后上下文帧
    tgt = np.stack([it["depth"].numpy() for it in items])
    msk = np.stack([it["mask"].numpy() for it in items])
    meta = [f"dom{it['domain']} ep{it['episode']} f{int(it['frame_id'][C - 1])}"
            for it in items]

    n_col = 1 + 2 * K
    for pg, r0 in enumerate(range(0, n, per_page)):
        rows = min(per_page, n - r0)
        fig, axes = plt.subplots(rows, n_col,
                                 figsize=(2.0 * n_col, 1.9 * rows))
        axes = np.atleast_2d(axes)
        for rr in range(rows):
            j = r0 + rr
            axes[rr, 0].imshow(rgb0[j])
            axes[rr, 0].set_title(f"start {meta[j]}", fontsize=7)
            for i in range(K):
                t = C + i
                axes[rr, 1 + i].imshow(
                    np.where(msk[j][t] > 0, tgt[j][t], np.nan),
                    cmap="viridis")
                axes[rr, 1 + i].set_title(f"+{i + 1} target", fontsize=7)
                axes[rr, 1 + K + i].imshow(mu_pred[i][j], cmap="viridis")
                axes[rr, 1 + K + i].set_title(f"+{i + 1} pred", fontsize=7)
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        out = out_prefix.parent / f"{out_prefix.stem}_p{pg}.png"
        fig.savefig(out, dpi=90)
        if writer is not None:
            writer.add_figure(f"val/unroll_depth_p{pg}", fig, step)
        plt.close(fig)
    T.train()
    D.train()


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=None,
                    help="默认取 model.yaml stage1.batch")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=1000,
                    help="每多少步存 ckpt_step{step}.pt 全量检查点")
    ap.add_argument("--resume", type=Path, default=None,
                    help="全量检查点路径：加载模型+优化器+RNG 继续训练")
    ap.add_argument("--overfit-batch", action="store_true",
                    help="单 batch 过拟合 sanity（每阶段训练前的硬规定）")
    ap.add_argument("--cache-dir", default=None,
                    help="默认 model.yaml stage1.cache_dir")
    ap.add_argument("--stage0-ckpt", type=Path,
                    default=Path("outputs/stage0/last.pt"),
                    help="初始化 D 与信号头（T/adapter 从零）")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/stage1"))
    ap.add_argument("--model-cfg", default="configs/model.yaml")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    model_cfg = yaml.safe_load(open(args.model_cfg))
    s1 = model_cfg["stage1"]
    C, K = int(s1["context_frames"]), int(s1["unroll_steps"])
    batch = args.batch or int(s1["batch"])
    lr = args.lr or float(s1["lr"])
    cache_dir = args.cache_dir or s1["cache_dir"]
    device = "cuda"

    train_ds = LatentClipDataset(cache_dir, context=C, horizon=K,
                                 split="train", stride=s1["clip_stride"])
    val_ds = LatentClipDataset(cache_dir, context=C, horizon=K, split="val",
                               stride=s1["clip_stride"])
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    print(f"clips: train={len(train_ds)} val={len(val_ds)} "
          f"(C={C}, K={K}, L={C + K})")

    tcfg = model_cfg["transition"]
    T = TransitionModel(d=tcfg["d"], depth=tcfg["depth"],
                        n_heads=tcfg["n_heads"],
                        grid_size=model_cfg["latent"]["grid_size"],
                        n_cond=tcfg["n_cond"], max_frames=tcfg["max_frames"],
                        n_reg=tcfg["n_reg"]).to(device)
    adapter = ActionAdapter(tcfg["action_dim"], tcfg["proprio_dim"],
                            d=tcfg["d"]).to(device)
    D = DepthDecoder(**model_cfg["depth_decoder"]).to(device)
    roles = model_cfg["encoder"]["register_roles"]
    heads = build_signals(model_cfg, dim=model_cfg["latent"]["dim"],
                          n_reg=tcfg["n_reg"]).to(device)
    # D 与信号头从 Stage 0 checkpoint 初始化（E 不进本循环）
    if args.stage0_ckpt.is_file():
        # 自己的 Stage 0 checkpoint（含 args 里的 PosixPath），可信源
        ck = torch.load(args.stage0_ckpt, map_location="cpu",
                        weights_only=False)
        D.load_state_dict(ck["D"])
        heads.load_state_dict(ck["signals"])
        print(f"D/signals <- {args.stage0_ckpt} (step {ck.get('step')})")
    else:
        print(f"警告：{args.stage0_ckpt} 不存在，D/信号头从零初始化")
    # 探针：挂预测输出，独立小优化器，特征 detach 不反传进 T
    probe = DomainProbe(dim=model_cfg["latent"]["dim"],
                        grid_size=model_cfg["latent"]["grid_size"]).to(device)
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    loss_cfg = dict(model_cfg["loss"])
    n_grad_scales = loss_cfg.pop("n_grad_scales")
    loss_cfg["lambda_teacher"] = 0.0        # 与 Stage 0 相同：mask-only 基线
    loss_fn = DepthLoss(n_grad_scales=n_grad_scales, **loss_cfg)

    opt = torch.optim.AdamW(
        [{"params": T.parameters(), "lr": lr},
         {"params": adapter.parameters(), "lr": lr},
         {"params": D.parameters(), "lr": lr},
         {"params": heads.parameters(), "lr": lr}],
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
            {"T": T, "adapter": adapter, "D": D, "signals": heads,
             "probe": probe},
            {"main": opt, "probe": probe_opt}, device)
        print(f"resume <- {args.resume}，从 step {start_step} 继续"
              f"（数据从新 epoch 洗牌，不逐 batch 复现原顺序）")
        if start_step >= args.steps:
            raise SystemExit(
                f"resume step {start_step} >= --steps {args.steps}，无需训练")
    step = start_step
    done = False
    single_batch = None
    while not done:
        for b in train_loader:
            if args.overfit_batch:
                if single_batch is None:
                    single_batch = b
                b = single_batch
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = rollout(T, adapter, b, C, K, device)
                loss, summary, probe_losses = compute_losses(
                    preds, b, C, D, heads, roles, loss_fn, s1,
                    device, probe=probe)
            (loss / args.accum).backward()

            # 探针独立优化器（仪器，不反传进 T——probe 内部 detach）
            for pl in probe_losses:
                (sum(pl.values()) / args.accum).backward()
            probe_opt.step()
            probe_opt.zero_grad(set_to_none=True)

            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(T.parameters()) + list(adapter.parameters())
                    + list(D.parameters()) + list(heads.parameters()), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % 10 == 0:
                rec = {"step": step, "elapsed_s": round(time.time() - t0, 1),
                       **{k: round(v.float().item(), 5)
                          for k, v in summary.items()}}
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if writer is not None:
                    for k, v in summary.items():
                        writer.add_scalar(f"train/{k}", v.float().item(), step)
                print(rec)
            if step % args.val_every == 0 or step == args.steps - 1:
                va = validate(T, adapter, D, heads, roles, loss_fn, probe,
                              val_loader, C, K, device, s1)
                if writer is not None:
                    for k, v in va.items():
                        writer.add_scalar(f"val/{k}", v, step)
                print(f"[val @ {step}] " + json.dumps(
                    {k: round(v, 5) for k, v in va.items()}))
                save_viz(T, adapter, D, val_ds, C, K, device,
                         args.outdir / f"viz_step{step:06d}.png",
                         writer=writer, step=step)
                torch.save(full_ckpt(step,
                                     {"T": T, "adapter": adapter, "D": D,
                                      "signals": heads, "probe": probe},
                                     {"main": opt, "probe": probe_opt}, args),
                           args.outdir / "last.pt")
            if step > 0 and step % args.save_every == 0:
                torch.save(full_ckpt(step,
                                     {"T": T, "adapter": adapter, "D": D,
                                      "signals": heads, "probe": probe},
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
