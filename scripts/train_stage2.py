#!/usr/bin/env python
"""Stage 2 训练：E+T 联合在线微调 + 域对抗（GRL 消灭 reg_agnostic 域信息）。

设计依据：guideline v2 §6-Stage 2 + 2026-09-16 用户裁决：
- **E+T 联合在线微调**：E 继续训练 ⇒ latent 缓存过期，clip 直接从原始
  HDF5 在线编码（MotionThinnedClipDataset，与 Stage 1 同 clip 语义：
  上下文 2 帧、自由展开 K=4、不跨 episode、sim 抽帧到 15fps）；
- **GRL 只挂 reg_agnostic**（register 分区纪律）：E 侧（全部 6 帧）与
  T 侧（r̂ 每展开步）双挂点（§2.2：T 递推 register ⇒ ẑ 侧同挂）；
  patch / reg_domain 只探针监控、不对抗（隔离优于摧毁）；
- **锚缩到 patch + reg_domain**（2026-09-16 裁决）：reg_agnostic 不再锚定
  （锚拉向带域信息的教师 = 与 GRL 直接拔河），让 GRL 自由清理；
- λ 从 0 在 5k step 内 ramp 到 0.1（model.yaml discriminator 节，禁止一步
  加满）；GRL batch 精确 50/50 域均衡（DomainBalancedBatchSampler，§9.1）；
- [2026-09-17 裁决] run1（λ=0.1/ramp 5k/判别器单步更新）复盘失败：
  E 在 GRL 压力下漂移过快，判别器追不上 ⇒ train disc acc 钉 0.5 是假象，
  离线探针 reg_agnostic 仍 1.000 线性可分（minimax 失衡）。修复：
  λ_max 0.1→0.03、ramp 5k→10k、判别器 lr 1e-4→3e-4、判别器内循环
  disc_inner_steps=4（每步在 detach 特征上额外更新，让判别器贴近当前 E）；
- [2026-09-17 裁决·二] run2 复盘：判别器过强（step 10 即 acc=1.0），
  GRL+BCE 在 sigmoid 饱和区 ⇒ 对抗梯度死亡，E 不受任何清洗压力，
  离线探针仍 1.000。修复：**non-saturating 对抗**（GAN 式）——E/T 侧
  改为判别器参数冻结图上的标签翻转 BCE（判别器越自信正确，E/T 梯度
  越大），判别器训练全部收进 disc_inner_loop（detach 特征，唯一通路）；
- [2026-09-17 裁决·三] run3 复盘：标签翻转 BCE 最优点是"判别器自信
  判错"，无稳定不动点 ⇒ acc 0↔1 overshoot 振荡，离线探针仍 0.999。
  路线切换：**HSIC（默认开，无对弈，跨步 FIFO 缓冲）+ 熵混淆（默认关，
  推向批次经验先验 π̂ 而非写死 0.5）独立开关**（model.yaml stage2.adv）；
  **val 探针独立**——validate() 现场从零训线性探针（不再借用 train
  阶段在线探针），逐域分层取样（修复历史缺陷：val_loader 顺序取 batch
  只拿到 real 单域），验收基线 = 经验先验 val/probe_prior。
  完整复盘见 src/sawvla/train_failed_log.md；
- 其余损失全部保留（全程不变量）：深度（真值帧梯度进 E+D；ẑ 侧 stop-grad
  只训 D，选项 A 延续）、L_dyn（patch+register）、信号（真值侧进 E、
  预测侧进 T）；
- 初始化：E <- Stage 0 ckpt；T/adapter/D/signals <- Stage 1 ckpt；
  判别器/探针从零；锚 = 冻结的原始 DINOv2 副本；
- checkpoint/恢复与 Stage 0/1 同机制（--save-every / --resume，
  优化器+RNG 全量）。

用法：
    python scripts/train_stage2.py --steps 200 --workers 4          # 冒烟
    python scripts/train_stage2.py --steps 20000 --batch 4 --workers 4
    python scripts/train_stage2.py --overfit-batch                  # sanity
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
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sawvla.data import (DomainBalancedBatchSampler,  # noqa: E402
                         MotionThinnedClipDataset, Stage0MotionThinnedDataset,
                         episode_split_tag)
from sawvla.data.image import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from sawvla.losses import (DepthLoss, HSICBuffer,  # noqa: E402
                           latent_prediction_loss, normalized_hsic)
from sawvla.models import (ActionAdapter, DINOv2Encoder, DomainDiscriminator,  # noqa: E402
                           DomainProbe, TransitionModel)
from sawvla.models import DepthDecoder  # noqa: E402
from sawvla.signals import SignalContext, build_signals  # noqa: E402

SUBSAMPLE = {"real": 1, "sim": 2}     # 2026-09-15 裁决：sim 30fps→15fps


# --------------------------------------------------------------------------- #
# 全量检查点（与 Stage 0/1 同机制）
# --------------------------------------------------------------------------- #

def full_ckpt(step: int, models: dict, opts: dict, args) -> dict:
    return {**{k: m.state_dict() for k, m in models.items()},
            **{f"opt/{k}": o.state_dict() for k, o in opts.items()},
            "rng": {"cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state()},
            "step": step, "args": vars(args)}


def load_full_ckpt(path, models: dict, opts: dict, device) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)  # 自己的 ckpt
    for k, m in models.items():
        m.load_state_dict(ck[k])
    for k, o in opts.items():
        o.load_state_dict(ck[f"opt/{k}"])
        for st in o.state.values():
            for kk, v in st.items():
                if torch.is_tensor(v):
                    st[kk] = v.to(device)
    if "rng" in ck:
        torch.set_rng_state(ck["rng"]["cpu"])
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        np.random.set_state(ck["rng"]["numpy"])
    return ck["step"] + 1


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #

def build_clip_sets(data_cfg, C, K, stride, motion_thresh, seed, val_frac):
    """两域 clip 数据集 + 各自 train/val 窗口划分（episode 级 crc32，同
    Stage 0/1 规则）。返回 (train_concat, val_concat, val_per_domain)——
    val 逐域列表用于分层取样（ConcatDataset 顺序取 batch 会只拿到 real，
    2026-09-17 发现的历史测量缺陷）。"""
    trains, vals = [], []
    for tag in ("real", "sim"):
        dom = data_cfg["domains"][tag]
        base = Stage0MotionThinnedDataset(
            root=dom["root"], domain_id=dom["domain_id"], camera=dom["camera"],
            depth_range=tuple(dom["depth_range"]), tasks=dom.get("tasks"),
            motion_thresh=motion_thresh, subsample=SUBSAMPLE[tag],
            verbose=False)
        keep_tr = {i for i, e in enumerate(base.episodes)
                   if episode_split_tag(e["name"], seed, val_frac) == "train"}
        keep_va = set(range(len(base.episodes))) - keep_tr
        trains.append(MotionThinnedClipDataset(base, C, K, stride,
                                               episode_keep=keep_tr))
        vals.append(MotionThinnedClipDataset(base, C, K, stride,
                                             episode_keep=keep_va))
    return ConcatDataset(trains), ConcatDataset(vals), vals


# --------------------------------------------------------------------------- #
# 前向与损失
# --------------------------------------------------------------------------- #

def encode(E, b, device):
    """在线编码 clip 全部帧：(B,L,...) → 三通道（带梯度，E 在训）。"""
    rgb = b["rgb"].to(device, non_blocking=True)
    B, L = rgb.shape[:2]
    tok = E.forward_tokens(rgb.flatten(0, 1))
    d = tok["patch"].shape[-1]
    return (tok["patch"].float().view(B, L, -1, d),
            tok["registers"].float().view(B, L, 4, d),
            tok["cls"].float().view(B, L, d))


def rollout(T, adapter, patch, regs, actions, proprio, C, K):
    """与 Stage 1 同语义，但输入带梯度（联合微调）。"""
    B, L = patch.shape[:2]
    act_tok, prop_tok = adapter(actions[:, :L - 1], proprio[:, :L - 1])
    z_in, r_in = patch[:, :C], regs[:, :C]
    pv = torch.ones(B, C, dtype=torch.bool, device=patch.device)
    preds = []
    for j in range(C, L):
        out = T.forward_with_registers(
            z_in, r_in, act_tok[:, :j], prop_tok[:, :j], pv, frame_offset=0)
        preds.append((out["patch"][:, -1], out["registers"][:, -1]))
        if j < L - 1:
            z_in = torch.cat([z_in, out["patch"][:, -1:]], dim=1)
            r_in = torch.cat([r_in, out["registers"][:, -1:]], dim=1)
            pv = torch.cat([pv, torch.zeros(B, 1, dtype=torch.bool,
                                            device=patch.device)], dim=1)
    return preds


def compute_losses(patch, regs, cls, preds, b, C, models, fns, s2, step):
    """全部损失 + 日志量。models/fns 见 main 装配。"""
    E, T, D, heads, anchor = (models[k] for k in
                              ("E", "T", "D", "heads", "anchor"))
    disc = models.get("disc")                    # 仅熵混淆开关打开时存在
    loss_fn, roles = fns["depth"], fns["roles"]
    reg_w = fns["reg_dyn_weight"]
    ag, dom = roles["agnostic"], roles["domain"]
    device = patch.device
    B, L = patch.shape[:2]
    K = len(preds)
    domain = b["domain"].to(device)
    domain_fl = domain.repeat_interleave(L)              # (B*L,)

    parts: dict[str, torch.Tensor] = {}
    loss = 0.0

    # --- L_dyn：patch + register，目标 sg（联合微调：梯度也经上下文进 E） ---
    for i, (z_hat, r_hat) in enumerate(preds):
        t = C + i
        parts[f"dyn/patch_{i}"] = latent_prediction_loss(
            z_hat, patch[:, t].detach(), z_prev=patch[:, t - 1].detach())
        parts[f"dyn/reg_{i}"] = latent_prediction_loss(
            r_hat, regs[:, t].detach(), z_prev=regs[:, t - 1].detach())
        loss = loss + parts[f"dyn/patch_{i}"] + reg_w * parts[f"dyn/reg_{i}"]

    # --- 深度：真值帧梯度进 E+D（全程不变量）；ẑ 侧 stop-grad 只训 D ---
    depth = b["depth"].to(device).flatten(0, 1)
    mask = b["mask"].to(device).flatten(0, 1).float()
    rgb_fl = b["rgb"].to(device).flatten(0, 1)
    mu, ls = D(patch.flatten(0, 1))
    p_true = loss_fn(mu, ls, depth, mask, torch.zeros_like(mu), rgb_fl)
    parts["depth/true"] = p_true["total"]
    loss = loss + p_true["total"]
    d_pred = 0.0
    for i, (z_hat, _) in enumerate(preds):
        t = C + i
        mu_p, ls_p = D(z_hat.detach())
        p_p = loss_fn(mu_p, ls_p, b["depth"][:, t].to(device),
                      b["mask"][:, t].to(device).float(),
                      torch.zeros_like(mu_p), b["rgb"][:, t].to(device))
        d_pred = d_pred + p_p["total"]
    parts["depth/pred"] = d_pred / K
    loss = loss + 0.5 * parts["depth/pred"]

    # --- 信号：真值侧（全帧，梯度进 E）；预测侧（选项 A：梯度进 T） ---
    if len(heads.heads) > 0:
        ctx_true = SignalContext(cls=cls.flatten(0, 1),
                                 registers=regs.flatten(0, 1),
                                 patch=patch.flatten(0, 1), roles=roles)
        for k_, v in heads.losses(
                ctx_true,
                {"proprio": b["proprio"].to(device).flatten(0, 1)}).items():
            parts[f"signal_true/{k_}"] = v
            loss = loss + v
        sp = 0.0
        for i, (z_hat, r_hat) in enumerate(preds):
            ctx_hat = SignalContext(cls=torch.zeros_like(cls[:, 0]),
                                    registers=r_hat, patch=z_hat, roles=roles)
            for k_, v in heads.losses(
                    ctx_hat,
                    {"proprio": b["proprio"][:, C + i].to(device)}).items():
                sp = sp + v
        parts["signal_pred"] = sp / K
        loss = loss + parts["signal_pred"]

    # --- 防漂移锚：patch + reg_domain（2026-09-16 裁决：agnostic 让位 GRL） ---
    if anchor is not None:
        with torch.no_grad():
            tok_ref = anchor.forward_tokens(rgb_fl)
        ref = torch.cat([tok_ref["patch"],
                         tok_ref["registers"][:, dom]], dim=1).float()
        cur = torch.cat([patch.flatten(0, 1),
                         regs.flatten(0, 1)[:, dom]], dim=1)
        parts["anchor"] = F.mse_loss(cur, ref)
        loss = loss + fns["anchor_w"] * parts["anchor"]

    # --- 清洗目标（2026-09-17 裁决·三）：HSIC（默认开）+ 熵混淆（默认关），
    #     独立开关；配置在 model.yaml stage2.adv，规格见 train_failed_log.md ---
    feat_e = regs.flatten(0, 1)[:, ag]               # (B*L, 2, d)

    # HSIC：无对弈惩罚，梯度直接进 E/T；跨步 FIFO 缓冲（detach 统计基底，
    # 梯度只经当前 batch）——单 batch 48/32 样本估计噪声太大，缓冲为硬性要求
    hc = fns.get("hsic") or {}
    if hc.get("enabled"):
        lam_h = hc["weight"] * min(1.0, step / hc["ramp_steps"])
        parts["hsic/lambda"] = torch.tensor(lam_h)
        if lam_h > 0:
            xe, ye = hc["buf_e"].join(feat_e.flatten(1), domain_fl.float())
            parts["hsic/e"] = normalized_hsic(xe, ye)
            xt = torch.cat([r_hat[:, ag].flatten(1) for _, r_hat in preds])
            xt, yt = hc["buf_t"].join(xt, domain.float().repeat(K))
            parts["hsic/t"] = normalized_hsic(xt, yt)
            loss = loss + lam_h * (parts["hsic/e"] + parts["hsic/t"])

    # 熵混淆（默认关）：把判别器后验推向**批次经验先验 π̂**（用户裁决：
    # 不写死 p=0.5，数据集并不完美均衡）；最优点 p=π̂ 是稳定不动点，
    # 取代 run3 的标签翻转 BCE。判别器训练全部在主循环 disc_inner_loop。
    ec = fns.get("entropy") or {}
    if ec.get("enabled") and disc is not None:
        lambd = ec["weight"] * min(1.0, step / ec["ramp_steps"])
        parts["grl/lambda"] = torch.tensor(lambd)
        with torch.no_grad():                        # 判别器读数（拔河监控）
            logit_e = disc(feat_e)
            parts["disc/acc_e"] = ((logit_e > 0).float()
                                   == domain_fl.float()).float().mean()
            parts["disc/e"] = F.binary_cross_entropy_with_logits(
                logit_e, domain_fl.float())
            logit_t = disc(preds[-1][1][:, ag])
            parts["disc/acc_t"] = ((logit_t > 0).float()
                                   == domain.float()).float().mean()
        if lambd > 0:
            for p in disc.parameters():              # 对抗梯度只进 E/T
                p.requires_grad_(False)
            logit_e_g = disc(feat_e)
            adv_e = F.binary_cross_entropy_with_logits(
                logit_e_g, torch.full_like(logit_e_g,
                                           float(domain_fl.float().mean())))
            a_t = 0.0
            pi_t = float(domain.float().mean())
            for _, r_hat in preds:
                lg = disc(r_hat[:, ag])
                a_t = a_t + F.binary_cross_entropy_with_logits(
                    lg, torch.full_like(lg, pi_t))
            for p in disc.parameters():
                p.requires_grad_(True)
            parts["grl/e"] = adv_e                   # E 侧熵混淆（推向 π̂）
            parts["grl/t"] = a_t / K
            loss = loss + lambd * (parts["grl/e"] + parts["grl/t"])
    return loss, parts


def disc_inner_loop(disc, opt, feat_e, dom_e, feats_t, dom_t, n_steps: int):
    """判别器训练（detach 特征上的 BCE，梯度只进判别器）——non-saturating
    改造后，这是判别器的**唯一**训练通路（主损失里的对抗项在 disc 参数
    冻结的图上，不给 disc 梯度）。特征在函数内强制 detach。
    返回末次损失。"""
    feat_e = feat_e.detach()
    feats_t = [f.detach() for f in feats_t]
    last = None
    for _ in range(n_steps):
        l = F.binary_cross_entropy_with_logits(disc(feat_e), dom_e)
        for f in feats_t:
            l = l + F.binary_cross_entropy_with_logits(disc(f), dom_t)
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
        opt.step()
        last = float(l.detach())
    return last


# --------------------------------------------------------------------------- #
# 验证与可视化
# --------------------------------------------------------------------------- #

def fresh_probe_acc(feat, labels, iters=500, lr=1e-2, seed=0):
    """val 独立探针（2026-09-17 裁决·三）：每次 val 现场**从零**训练一个
    线性探针（与离线终审同配方），不借用 train 阶段在线探针——run1/run3
    证明在线探针会滞后/翻转，读数不可信。随机 50/50 分训练/评估两半。
    返回 (eval_acc, prior)：prior = 评估半的经验多数类比率（验收基线，
    不假设 0.5——数据集并不完美均衡）。"""
    n = feat.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).to(feat.device)
    tr, va = perm[: n // 2], perm[n // 2:]
    x = feat.flatten(1).float().detach()
    y = labels.long()
    with torch.enable_grad():            # validate 在 no_grad 里，探针训练要梯度
        w = torch.zeros(x.shape[1], 2, device=x.device, requires_grad=True)
        bias = torch.zeros(2, device=x.device, requires_grad=True)
        opt = torch.optim.Adam([w, bias], lr=lr)
        for _ in range(iters):
            l = F.cross_entropy(x[tr] @ w + bias, y[tr])
            opt.zero_grad()
            l.backward()
            opt.step()
        acc = ((x[va] @ w + bias).argmax(-1) == y[va]).float().mean().item()
    frac = y[va].float().mean().item()
    return acc, max(frac, 1.0 - frac)


@torch.no_grad()
def validate(models, fns, val_loaders, C, K, device, s2,
             max_batches: int = 10, probe_seed: int = 0):
    """验证：损失/逐步误差 + 独立探针。val_loaders = 逐域 loader 列表
    （分层各取 max_batches//2 批，两域都覆盖）；也可直接传"逐域 batch
    列表的列表"（ckpt 扫扫时让所有 ckpt 评同一批样本）。"""
    E, T, adapter, D, heads = (models[k] for k in
                               ("E", "T", "adapter", "D", "heads"))
    disc = models.get("disc")
    for m in (E, T, D):
        m.eval()
    roles = fns["roles"]
    per_dom = max(1, max_batches // len(val_loaders))
    batches = []
    for loader in val_loaders:
        for i, b in enumerate(loader):
            if i >= per_dom:
                break
            batches.append(b)
    tot, step_err, step_err_r, n = {}, None, None, 0
    pool = {"patch": [], "reg_domain": [], "reg_agnostic": []}
    labs = []
    for b in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch, regs, cls = encode(E, b, device)
            preds = rollout(T, adapter, patch, regs,
                            b["action"].to(device).float(),
                            b["proprio"].to(device).float(), C, K)
            _, parts = compute_losses(patch, regs, cls, preds, b, C,
                                      {"E": E, "T": T, "D": D, "heads": heads,
                                       "disc": disc, "anchor": None},
                                      fns, s2, step=0)
        for k, v in parts.items():
            if k.startswith(("disc/", "grl/", "hsic/")):
                continue
            tot[k] = tot.get(k, 0.0) + float(v)
        # 逐步 latent 误差
        se = [((ph - patch[:, C + j]) ** 2).mean().item()
              for j, (ph, _) in enumerate(preds)]
        se_r = [((rh - regs[:, C + j]) ** 2).mean().item()
                for j, (_, rh) in enumerate(preds)]
        step_err = se if step_err is None else [a + c for a, c in
                                                zip(step_err, se)]
        step_err_r = se_r if step_err_r is None else [a + c for a, c in
                                                      zip(step_err_r, se_r)]
        # 独立探针的特征池（冻结，三通道）
        pool["patch"].append(patch.flatten(0, 1).mean(1).float())
        pool["reg_domain"].append(
            regs.flatten(0, 1)[:, roles["domain"]].flatten(1).float())
        pool["reg_agnostic"].append(
            regs.flatten(0, 1)[:, roles["agnostic"]].flatten(1).float())
        labs.append(b["domain"].to(device).repeat_interleave(
            patch.shape[1]).float())
        n += 1
    for m in (E, T, D):
        m.train()
    out = {k: v / max(n, 1) for k, v in tot.items()}
    for j in range(K):
        out[f"latent_mse/step_{j + 1}"] = step_err[j] / max(n, 1)
        out[f"latent_mse_reg/step_{j + 1}"] = step_err_r[j] / max(n, 1)
    y = torch.cat(labs)
    for ch, feats in pool.items():
        acc, prior = fresh_probe_acc(torch.cat(feats), y, seed=probe_seed)
        out[f"probe_e/{ch}"] = acc
        out["probe_prior"] = prior
    return out


@torch.no_grad()
def save_viz(models, val_ds, C, K, device, out_prefix: Path, writer=None,
             step: int = 0, n_clips: int = 100, per_page: int = 25):
    """与 Stage 1 同版式（起点 rgb | +1..K target 集中 | +1..K pred 集中），
    在线 E 编码。以 step 为种子抽样。"""
    E, T, adapter, D = (models[k] for k in ("E", "T", "adapter", "D"))
    E.eval(); T.eval(); D.eval()
    rng = np.random.default_rng(step)
    idx = rng.choice(len(val_ds), size=min(n_clips, len(val_ds)),
                     replace=False)
    items = [val_ds[int(i)] for i in idx]
    mu_pred = [[] for _ in range(K)]
    vbatch = 16
    for b0 in range(0, len(items), vbatch):
        b = torch.utils.data.default_collate(items[b0: b0 + vbatch])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch, regs, _ = encode(models["E"], b, device)
            preds = rollout(T, adapter, patch, regs,
                            b["action"].to(device).float(),
                            b["proprio"].to(device).float(), C, K)
        for i, (z_hat, _) in enumerate(preds):
            mu, _ = D(z_hat.detach())
            mu_pred[i].append(mu.float().cpu().numpy())
    mu_pred = [np.concatenate(m) for m in mu_pred]
    n = len(items)
    rgb0 = np.stack([(it["rgb"][C - 1].numpy().transpose(1, 2, 0)
                      * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
                     for it in items])
    tgt = np.stack([it["depth"].numpy() for it in items])
    msk = np.stack([it["mask"].numpy() for it in items])
    meta = [f"dom{it['domain']} ep{it['episode_id']} "
            f"f{int(it['frame_id'][C - 1])}" for it in items]
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
                    np.where(msk[j][t] > 0, tgt[j][t], np.nan), cmap="viridis")
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
    E.train(); T.train(); D.train()


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--motion-thresh", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--overfit-batch", action="store_true")
    ap.add_argument("--stage0-ckpt", type=Path,
                    default=Path("outputs/stage0/last.pt"))
    ap.add_argument("--stage1-ckpt", type=Path,
                    default=Path("outputs/stage1/last.pt"))
    ap.add_argument("--outdir", type=Path, default=Path("outputs/stage2"))
    ap.add_argument("--model-cfg", default="configs/model.yaml")
    ap.add_argument("--data-cfg", default="configs/data.yaml")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    model_cfg = yaml.safe_load(open(args.model_cfg))
    data_cfg = yaml.safe_load(open(args.data_cfg))
    s1, s2 = model_cfg["stage1"], model_cfg["stage2"]
    C, K = int(s1["context_frames"]), int(s1["unroll_steps"])
    batch = args.batch or int(s2["batch"])
    lr = args.lr or float(s2["lr"])
    device = "cuda"

    train_cat, val_cat, val_per_dom = build_clip_sets(
        data_cfg, C, K, s2["clip_stride"], args.motion_thresh, args.seed, 0.1)
    n_real = len(train_cat.datasets[0])
    sampler = DomainBalancedBatchSampler(n_real, len(train_cat) - n_real,
                                         batch, seed=args.seed)
    train_loader = DataLoader(train_cat, batch_sampler=sampler,
                              num_workers=args.workers, pin_memory=True)
    # 逐域 val loader：分层取样，两域都覆盖（ConcatDataset 顺序取 batch
    # 会只拿到 real——run1-3 的 val 探针/指标其实只测了单域，已记入台账）；
    # shuffle=True（共享 generator 逐 epoch 推进）：每次 validate 的前几批
    # 不再固定落在同一批 episode 上（run4 缺陷：顺序取批 ⇒ 每域只见 1 个
    # episode，探针测的是"这一集 vs 那一集"而非域信息）。
    val_gen = torch.Generator().manual_seed(args.seed)
    val_loaders = [DataLoader(v, batch_size=batch, shuffle=True,
                              generator=val_gen,
                              num_workers=args.workers, pin_memory=True)
                   for v in val_per_dom]
    print(f"clips: train={len(train_cat)} (real {n_real} / sim "
          f"{len(train_cat) - n_real}), val={len(val_cat)}, batch={batch} 50/50")

    # --- 模型装配 ---
    E = DINOv2Encoder(model_cfg["paths"]["encoder"]).to(device)
    ck0 = torch.load(args.stage0_ckpt, map_location="cpu", weights_only=False)
    E.load_state_dict(ck0["E"])
    anchor = DINOv2Encoder(model_cfg["paths"]["encoder"]).to(device)
    anchor.freeze()
    anchor.eval()
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
    ck1 = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    T.load_state_dict(ck1["T"])
    adapter.load_state_dict(ck1["adapter"])
    D.load_state_dict(ck1["D"])
    heads.load_state_dict(ck1["signals"])
    print(f"E <- {args.stage0_ckpt} (step {ck0.get('step')}); "
          f"T/adapter/D/signals <- {args.stage1_ckpt} (step {ck1.get('step')})")
    # 清洗开关（2026-09-17 裁决·三）：HSIC / 熵混淆独立开关，默认开 HSIC。
    # 判别器只在熵混淆开启时建造（省算力；HSIC 无对弈不需要判别器）。
    adv_cfg = s2.get("adv", {})
    hsic_cfg = dict(adv_cfg.get("hsic", {}))
    if hsic_cfg.get("enabled"):
        cap = int(hsic_cfg.get("buffer_size", 256))
        hsic_cfg["weight"] = float(hsic_cfg["weight"])
        hsic_cfg["ramp_steps"] = float(hsic_cfg["ramp_steps"])
        hsic_cfg["buf_e"] = HSICBuffer(cap)
        hsic_cfg["buf_t"] = HSICBuffer(cap)
    ent_cfg = dict(adv_cfg.get("entropy", {}))
    ent_on = bool(ent_cfg.get("enabled"))
    if ent_on:
        ent_cfg["weight"] = float(ent_cfg["weight"])
        ent_cfg["ramp_steps"] = float(ent_cfg["ramp_steps"])
    dcfg = model_cfg["discriminator"]
    if ent_on:
        # 判别器：register 扁平模式（in_tokens=2，只读 reg_agnostic）
        disc = DomainDiscriminator(d=model_cfg["latent"]["dim"],
                                   hidden=dcfg["hidden"],
                                   in_tokens=len(roles["agnostic"])).to(device)
        disc_inner = int(dcfg.get("disc_inner_steps", 4))
        disc_opt = torch.optim.AdamW(disc.parameters(),
                                     lr=float(dcfg.get("disc_lr", 3e-4)))
    else:
        disc = disc_opt = None
        disc_inner = 0
    probe = DomainProbe(dim=model_cfg["latent"]["dim"],
                        grid_size=model_cfg["latent"]["grid_size"]).to(device)
    print(f"adv: hsic={'on' if hsic_cfg.get('enabled') else 'off'} "
          f"(w={hsic_cfg.get('weight')}, ramp={hsic_cfg.get('ramp_steps')}, "
          f"buf={hsic_cfg.get('buffer_size')}) | entropy="
          f"{'on' if ent_on else 'off'}")

    loss_cfg = dict(model_cfg["loss"])
    n_grad_scales = loss_cfg.pop("n_grad_scales")
    loss_cfg["lambda_teacher"] = 0.0
    loss_fn = DepthLoss(n_grad_scales=n_grad_scales, **loss_cfg)
    fns = {"depth": loss_fn, "roles": roles,
           "reg_dyn_weight": float(s1["reg_dyn_weight"]),
           "hsic": hsic_cfg, "entropy": ent_cfg,
           "anchor_w": (float(s2["anchor"]["distill_weight"]))}

    opt = torch.optim.AdamW(
        [{"params": E.parameters(), "lr": lr},
         {"params": T.parameters(), "lr": lr},
         {"params": adapter.parameters(), "lr": lr},
         {"params": D.parameters(), "lr": lr},
         {"params": heads.parameters(), "lr": lr}], weight_decay=0.01)
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    log_path = args.outdir / "log.jsonl"
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.outdir / "tb")
        print(f"tensorboard: tensorboard --logdir {args.outdir / 'tb'}")
    except ImportError:
        print("警告：未安装 tensorboard，标量只写 log.jsonl")
        writer = None

    models = {"E": E, "T": T, "adapter": adapter, "D": D, "heads": heads,
              "anchor": anchor}
    if disc is not None:
        models["disc"] = disc
    opts = {"main": opt, "probe": probe_opt}
    if disc_opt is not None:
        opts["disc"] = disc_opt
    t0 = time.time()
    start_step = 0
    if args.resume:
        start_step = load_full_ckpt(
            args.resume, {**models, "probe": probe}, opts, device)
        print(f"resume <- {args.resume}，从 step {start_step} 继续")
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
                patch, regs, cls = encode(E, b, device)
                preds = rollout(T, adapter, patch, regs,
                                b["action"].to(device).float(),
                                b["proprio"].to(device).float(), C, K)
                loss, parts = compute_losses(patch, regs, cls, preds, b, C,
                                             models, fns, s2, step)
                # 探针：E 侧三通道 + ẑ 侧（detach 在 probe 内部）
                ctx_e = SignalContext(cls=cls.flatten(0, 1),
                                      registers=regs.flatten(0, 1),
                                      patch=patch.flatten(0, 1), roles=roles)
                dom_fl = b["domain"].to(device).repeat_interleave(
                    patch.shape[1])
                probe_losses = probe.losses(ctx_e, dom_fl)
            loss.backward()

            for pl_ in [sum(probe_losses.values())]:
                pl_.backward()
            probe_opt.step()
            probe_opt.zero_grad(set_to_none=True)
            torch.nn.utils.clip_grad_norm_(
                list(E.parameters()) + list(T.parameters())
                + list(adapter.parameters()) + list(D.parameters())
                + list(heads.parameters()), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            # 判别器训练（仅熵混淆开启时；detach 特征，唯一通路）
            if disc is not None:
                inner_loss = disc_inner_loop(
                    disc, disc_opt,
                    regs.flatten(0, 1)[:, roles["agnostic"]],
                    b["domain"].to(device).repeat_interleave(
                        patch.shape[1]).float(),
                    [rh[:, roles["agnostic"]] for _, rh in preds],
                    b["domain"].to(device).float(), max(1, disc_inner))
                parts["disc/inner"] = torch.tensor(inner_loss)
            # HSIC 跨步缓冲：每步推入当前 batch 的 detach 特征（λ=0 的 ramp
            # 期也推，保证 λ 生效时统计基底已满）
            if hsic_cfg.get("enabled"):
                hsic_cfg["buf_e"].push(
                    regs.flatten(0, 1)[:, roles["agnostic"]].flatten(1),
                    b["domain"].to(device).repeat_interleave(
                        patch.shape[1]).float())
                hsic_cfg["buf_t"].push(
                    torch.cat([rh[:, roles["agnostic"]].flatten(1)
                               for _, rh in preds]),
                    b["domain"].to(device).float().repeat(K))
            # train 流在线探针：只作参考读数（验收以 val 独立探针为准）
            if step % 10 == 0:
                with torch.no_grad():
                    for k, v in probe.accuracies(ctx_e, dom_fl).items():
                        parts[f"probe/{k}"] = torch.tensor(float(v))

            if step % 10 == 0:
                rec = {"step": step, "elapsed_s": round(time.time() - t0, 1),
                       **{k: round(float(v), 5) for k, v in parts.items()}}
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if writer is not None:
                    for k, v in parts.items():
                        writer.add_scalar(f"train/{k}", float(v), step)
                print(rec)
            if step % args.val_every == 0 or step == args.steps - 1:
                va = validate(models, fns, val_loaders, C, K, device, s2)
                if writer is not None:
                    for k, v in va.items():
                        writer.add_scalar(f"val/{k}", v, step)
                print(f"[val @ {step}] " + json.dumps(
                    {k: round(v, 5) for k, v in va.items()}))
                save_viz(models, val_cat, C, K, device,
                         args.outdir / f"viz_step{step:06d}.png",
                         writer=writer, step=step)
                torch.save(full_ckpt(step, {**models, "probe": probe},
                                     opts, args),
                           args.outdir / "last.pt")
            if step > 0 and step % args.save_every == 0:
                torch.save(full_ckpt(step, {**models, "probe": probe},
                                     opts, args),
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
