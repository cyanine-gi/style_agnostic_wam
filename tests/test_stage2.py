"""Stage 2 单测：判别器扁平模式、在线 clip 数据集、50/50 域均衡采样、
GRL/锚的分区纪律、compute_losses 联合损失通路。

不依赖真实数据与真实权重（tiny 合成张量 + 合成 HDF5 episode）。
"""

import numpy as np
import pytest
import torch

from sawvla.data import (DomainBalancedBatchSampler,
                         MotionThinnedClipDataset, Stage0MotionThinnedDataset)
from sawvla.losses import (DepthLoss, HSICBuffer, grad_reverse,  # noqa: F401
                           latent_prediction_loss, normalized_hsic)
from sawvla.models import DepthDecoder, DomainDiscriminator
from sawvla.signals import SignalContext, build_signals
from test_dataset import make_episode

D, G = 32, 4
ROLES = {"domain": [0, 1], "agnostic": [2, 3]}


# --------------------------------------------------------------------------- #
# 判别器扁平（register）模式
# --------------------------------------------------------------------------- #

def test_disc_flat_mode_shape_and_capacity():
    disc = DomainDiscriminator(d=D, hidden=16, in_tokens=2)
    assert disc(torch.randn(4, 2, D)).shape == (4,)
    with pytest.raises(ValueError, match="in_tokens"):
        disc(torch.randn(4, 3, D))
    assert sum(p.numel() for p in disc.parameters()) / 1e6 < 2.0


def test_disc_flat_mode_grl_sign():
    """扁平模式下 GRL 梯度符号同样反转（§11.3-3）。"""
    torch.manual_seed(0)
    disc = DomainDiscriminator(d=D, hidden=16, in_tokens=2)
    x = torch.randn(2, 2, D, requires_grad=True)
    g_plain = torch.autograd.grad(disc(x).sum(), x, retain_graph=True)[0]
    g_grl = torch.autograd.grad(disc(grad_reverse(x, 0.5)).sum(), x)[0]
    assert torch.allclose(g_grl, -0.5 * g_plain, atol=1e-6)


def test_grl_partition_discipline():
    """GRL 只挂 reg_agnostic：梯度只能进 agnostic 列，domain 列零梯度。"""
    torch.manual_seed(0)
    disc = DomainDiscriminator(d=D, hidden=16, in_tokens=2)
    regs = torch.randn(3, 4, D, requires_grad=True)
    logit = disc(grad_reverse(regs[:, ROLES["agnostic"]], 0.1))
    logit.sum().backward()
    g = regs.grad
    assert g[:, ROLES["agnostic"]].abs().sum() > 0
    assert g[:, ROLES["domain"]].abs().sum() == 0


# --------------------------------------------------------------------------- #
# 清洗目标（2026-09-17 裁决·三）：HSIC + 熵混淆（先验目标）+ val 独立探针
# --------------------------------------------------------------------------- #

def test_normalized_hsic_independent_vs_dependent():
    """HSIC：独立特征 ≈0，域依赖特征显著大；梯度可回传特征。"""
    torch.manual_seed(0)
    y = torch.randint(0, 2, (200,)).float()
    x_ind = torch.randn(200, 16)
    x_dep = x_ind.clone()
    x_dep[y.bool()] += 3.0
    h_ind = float(normalized_hsic(x_ind, y))
    h_dep = float(normalized_hsic(x_dep, y))
    assert h_dep > 5 * max(h_ind, 1e-6), f"依赖应远大于独立: {h_dep} vs {h_ind}"
    assert h_ind < 0.1
    x = x_dep.clone().requires_grad_(True)
    normalized_hsic(x, y).backward()
    assert x.grad.abs().sum() > 0


def test_hsic_buffer_fifo_and_grad_discipline():
    """跨步 FIFO 缓冲：容量截断、join 拼接、当前 batch 带梯度、缓冲永远 detach。"""
    buf = HSICBuffer(capacity=8)
    buf.push(torch.randn(5, 4), torch.zeros(5))
    buf.push(torch.randn(5, 4), torch.ones(5))
    assert buf.x.shape == (8, 4)                     # FIFO 截断到容量
    x = torch.randn(3, 4, requires_grad=True)
    xj, yj = buf.join(x, torch.zeros(3))
    assert xj.shape == (11, 4) and yj.shape == (11,)
    xj.sum().backward()
    assert x.grad.abs().sum() > 0                    # 当前 batch 带梯度
    assert not buf.x.requires_grad                   # 缓冲永远 detach


def test_entropy_confusion_targets_batch_prior():
    """熵混淆目标 = 批次经验先验 π̂（不写死 0.5）：软标签 BCE 在 p=π̂ 取最小。"""
    pi = 0.71                                        # 数据集自然比率 ≈71:29
    logit = torch.linspace(-4, 4, 401)
    tgt = torch.full_like(logit, pi)
    l = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, tgt, reduction="none")
    best_p = torch.sigmoid(logit[l.argmin()]).item()
    assert abs(best_p - pi) < 0.03


def test_fresh_probe_acc_separable_and_prior():
    """val 独立探针：可分特征 ⇒ 高准确率；prior = 评估半经验多数类比率。"""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve()
                        .parent.parent / "scripts"))
    from train_stage2 import fresh_probe_acc

    torch.manual_seed(0)
    y = torch.cat([torch.zeros(120), torch.ones(40)])    # 不均衡 3:1
    x = torch.randn(160, 8)
    x[y.bool()] += 3.0
    acc, prior = fresh_probe_acc(x, y, iters=300)
    assert acc > 0.95
    assert 0.6 < prior < 0.9                             # ≈0.75


def test_disc_inner_loop_learns_and_no_leak():
    """2026-09-17 裁决的判别器内循环：detach 特征上可分 ⇒ 内循环学会；
    且梯度绝不回传给带来计算图的特征（不外泄到 E/T）。"""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve()
                        .parent.parent / "scripts"))
    from train_stage2 import disc_inner_loop

    torch.manual_seed(0)
    disc = DomainDiscriminator(d=D, hidden=16, in_tokens=2)
    opt = torch.optim.AdamW(disc.parameters(), lr=1e-2)
    dom_e = torch.randint(0, 2, (64,)).float()
    dom_t = torch.randint(0, 2, (16,)).float()
    feat_e = torch.randn(64, 2, D, requires_grad=True)
    feat_t = torch.randn(16, 2, D, requires_grad=True)
    with torch.no_grad():                      # 造线性可分的域偏移
        feat_e[dom_e.bool()] += 3.0
        feat_t[dom_t.bool()] += 3.0
    acc0 = ((disc(feat_e.detach()) > 0).float() == dom_e).float().mean()
    disc_inner_loop(disc, opt, feat_e, dom_e, [feat_t], dom_t, n_steps=30)
    acc1 = ((disc(feat_e.detach()) > 0).float() == dom_e).float().mean()
    assert acc1 > 0.95 and acc1 > acc0
    assert feat_e.grad is None and feat_t.grad is None   # 梯度不外泄


def test_anchor_scope_excludes_agnostic():
    """锚范围 patch+reg_domain：锚损失对 reg_agnostic 列零梯度。"""
    patch = torch.randn(2, G * G, D, requires_grad=True)
    regs = torch.randn(2, 4, D, requires_grad=True)
    cur = torch.cat([patch, regs[:, ROLES["domain"]]], dim=1)
    torch.nn.functional.mse_loss(cur, torch.zeros_like(cur)).backward()
    assert regs.grad[:, ROLES["domain"]].abs().sum() > 0
    assert regs.grad[:, ROLES["agnostic"]].abs().sum() == 0
    assert patch.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# 在线 clip 数据集
# --------------------------------------------------------------------------- #

def test_clip_dataset_online(tmp_path):
    make_episode(tmp_path, "task_a", "ep_001", n_frames=10, seed=1)
    make_episode(tmp_path, "task_a", "ep_002", n_frames=8, seed=2)
    base = Stage0MotionThinnedDataset(tmp_path, domain_id=0,
                                      camera="camera_front",
                                      motion_thresh=0.0, verbose=False)
    ds = MotionThinnedClipDataset(base, context=2, horizon=2)
    # span=4：ep1(10帧)→7 窗口，ep2(8帧)→5 窗口
    assert len(ds) == 7 + 5
    item = ds[0]
    assert item["rgb"].shape[0] == 4 and item["action"].shape == (4, 16)
    assert torch.equal(item["frame_id"][1:] - item["frame_id"][:-1],
                       torch.ones(3, dtype=torch.long))
    # episode_keep 过滤
    ds_v = MotionThinnedClipDataset(base, context=2, horizon=2,
                                    episode_keep={1})
    assert len(ds_v) == 5
    assert all(ds_v[i]["episode_id"] == 1 for i in range(len(ds_v)))
    base.close()


# --------------------------------------------------------------------------- #
# 50/50 域均衡采样
# --------------------------------------------------------------------------- #

def test_balanced_sampler_exact_5050():
    s = DomainBalancedBatchSampler(n_first=10, n_second=6, batch_size=4,
                                   seed=0)
    batches = list(iter(s))
    assert len(s) == 10 // 2             # 由较大域决定
    seen_second = 0
    for b in batches:
        assert len(b) == 4
        a = [i for i in b if i < 10]
        bb = [i for i in b if i >= 10]
        assert len(a) == 2 and len(bb) == 2          # 精确 50/50
        assert all(10 <= i < 16 for i in bb)
        seen_second += len(bb)
    assert seen_second >= 6              # 小域被覆盖（循环复用）


def test_balanced_sampler_odd_batch_rejected():
    with pytest.raises(ValueError, match="偶数"):
        DomainBalancedBatchSampler(10, 10, 3)


# --------------------------------------------------------------------------- #
# compute_losses 联合通路（tiny）
# --------------------------------------------------------------------------- #

def test_compute_losses_joint_backward():
    """tiny 装配跑一遍 compute_losses：损失有限，梯度到达 D/disc/heads/输入。"""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve()
                        .parent.parent / "scripts"))
    from train_stage2 import compute_losses

    torch.manual_seed(0)
    B, L, C, K = 2, 4, 2, 2
    Ddec = DepthDecoder(d=D, grid_size=G, out_size=G * 4, width1=16,
                        width2=16)
    heads = build_signals(
        {"latent": {"grid_size": G, "dim": D},
         "encoder": {"register_roles": ROLES},
         "signals": {"joint_pos": {"enabled": True, "weight": 1.0}}},
        dim=D, n_reg=4)
    disc = DomainDiscriminator(d=D, hidden=16, in_tokens=2)
    loss_fn = DepthLoss(lambda_metric=1.0, lambda_teacher=0.0,
                        lambda_grad=0.5, lambda_smooth=0.1, n_grad_scales=2)
    fns = {"depth": loss_fn, "roles": ROLES, "reg_dyn_weight": 1.0,
           "anchor_w": 0.1,
           "hsic": {"enabled": True, "weight": 1.0, "ramp_steps": 5000.0,
                    "buf_e": HSICBuffer(64), "buf_t": HSICBuffer(64)},
           "entropy": {"enabled": True, "weight": 0.03,
                       "ramp_steps": 5000.0}}
    models = {"E": None, "T": None, "D": Ddec, "heads": heads, "disc": disc,
              "anchor": None}

    patch = torch.randn(B, L, G * G, D, requires_grad=True)
    regs = torch.randn(B, L, 4, D, requires_grad=True)
    cls = torch.randn(B, L, D)
    preds = [(torch.randn(B, G * G, D, requires_grad=True),
              torch.randn(B, 4, D, requires_grad=True)) for _ in range(K)]
    b = {"depth": torch.rand(B, L, G * 4, G * 4) + 0.5,
         "mask": torch.ones(B, L, G * 4, G * 4),
         "rgb": torch.rand(B, L, 3, 32, 32),
         "proprio": torch.randn(B, L, 16),
         "domain": torch.tensor([0, 1])}

    loss, parts = compute_losses(patch, regs, cls, preds, b, C, models,
                                 fns, None, step=5000)   # λ 已 ramp 满
    assert torch.isfinite(loss)
    for k in ("depth/true", "depth/pred", "signal_pred", "grl/e", "grl/t",
              "hsic/e", "hsic/t", "hsic/lambda", "dyn/patch_0"):
        assert k in parts
    loss.backward()
    assert patch.grad.abs().sum() > 0          # dyn/深度/信号经上下文回传
    assert regs.grad[:, ROLES["agnostic"]].abs().sum() > 0   # HSIC+熵混淆+信号
    # non-saturating 改造后：主损失的对抗项在 disc 参数冻结的图上，
    # 判别器绝不能从主 backward 收梯度（唯一训练通路 = disc_inner_loop）
    assert all(p.grad is None for p in disc.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in Ddec.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in heads.parameters())
    # 预测侧信号梯度进 T 的 r̂（选项 A）
    assert preds[0][1].grad is not None and preds[0][1].grad.abs().sum() > 0
    # 深度对预测 patch stop-grad：preds 的 ẑ 只应收到 dyn 梯度（存在即可，
    # detach 正确性由 test_option_a_gradient_flow（Stage 1）钉死）
    assert preds[0][0].grad is not None
