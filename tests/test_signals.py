"""监督信号注册表 + 域探针单测（合成张量，不需要真实权重）。

覆盖：注册机制、装配、joint_pos 目标构造（剔夹爪）、分区纪律（梯度只进
域无关槽位）、探针不反传进 E。
"""

import pytest
import torch

from sawvla.models import DomainProbe
from sawvla.signals import (REGISTRY, SignalContext, SignalHead,
                            build_signals, register_signal)

DIM, N_REG = 32, 4
ROLES = {"domain": [0, 1], "agnostic": [2, 3]}
MODEL_CFG = {
    "latent": {"grid_size": 4, "dim": DIM},
    "encoder": {"register_roles": ROLES},
    "signals": {"joint_pos": {"enabled": True, "weight": 0.5}},
}


def make_ctx(requires_grad=False):
    g = torch.get_default_dtype()
    ctx = SignalContext(
        cls=torch.randn(2, DIM),
        registers=torch.randn(2, N_REG, DIM),
        patch=torch.randn(2, 16, DIM),
        roles=ROLES)
    if requires_grad:
        ctx.registers.requires_grad_(True)
        ctx.patch.requires_grad_(True)
    return ctx


def make_batch():
    return {"proprio": torch.randn(2, 16),
            "domain": torch.tensor([0, 1])}


def test_register_duplicate_rejected():
    with pytest.raises(KeyError, match="重复注册"):
        register_signal("joint_pos")(SignalHead)


def test_build_unknown_signal_rejected():
    bad = {**MODEL_CFG,
           "signals": {"nonexistent": {"enabled": True}}}
    with pytest.raises(KeyError, match="未注册"):
        build_signals(bad, dim=DIM, n_reg=N_REG)


def test_disabled_signal_skipped():
    cfg = {**MODEL_CFG,
           "signals": {"joint_pos": {"enabled": False}}}
    assert len(build_signals(cfg, dim=DIM, n_reg=N_REG).heads) == 0


def test_joint_pos_target_drops_grippers():
    """proprio 契约 [L7,R7,gL,gR]：目标必须恰好是 0:7 与 7:14，共 14 维。"""
    heads = build_signals(MODEL_CFG, dim=DIM, n_reg=N_REG)
    head = heads.heads[0]
    p = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    tgt = head.target({"proprio": p})
    assert tgt.shape == (2, 14)
    assert torch.equal(tgt, p[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    # 14=左夹爪、15=右夹爪不得出现（每行 arange 偏移 16，gL=14, gR=15）
    assert not torch.equal(tgt[:, -1], p[:, 15])


def test_joint_pos_loss_and_weight():
    heads = build_signals(MODEL_CFG, dim=DIM, n_reg=N_REG)
    losses = heads.losses(make_ctx(), make_batch())
    assert set(losses) == {"joint_pos"}
    assert losses["joint_pos"].ndim == 0 and torch.isfinite(losses["joint_pos"])


def test_partition_discipline_gradients():
    """分区纪律：joint_pos 的梯度只能进 reg_agnostic 列，reg_domain 必须为 0。"""
    heads = build_signals(MODEL_CFG, dim=DIM, n_reg=N_REG)
    ctx = make_ctx(requires_grad=True)
    sum(heads.losses(ctx, make_batch()).values()).backward()
    g = ctx.registers.grad
    assert g[:, ROLES["agnostic"]].abs().sum() > 0      # 域无关槽位有梯度
    assert g[:, ROLES["domain"]].abs().sum() == 0        # 域槽位零梯度
    assert ctx.patch.grad is None                        # patch 完全不被碰


def test_probe_does_not_backprop_into_encoder():
    """探针是仪器：它的损失反传后，encoder 特征不得有梯度。"""
    probe = DomainProbe(dim=DIM)
    ctx = make_ctx(requires_grad=True)
    sum(probe.losses(ctx, make_batch()["domain"]).values()).backward()
    assert ctx.registers.grad is None and ctx.patch.grad is None
    assert all(p.grad is not None for p in probe.parameters())


def test_probe_accuracy_interface():
    probe = DomainProbe(dim=DIM)
    accs = probe.accuracies(make_ctx(), make_batch()["domain"])
    assert set(accs) == {"patch", "reg_domain", "reg_agnostic"}
    assert all(0.0 <= v <= 1.0 for v in accs.values())


def test_token_probe_shapes():
    """逐 token 热图探针：逐位置准确率 (N,)，取值 [0,1]。"""
    probe = DomainProbe(dim=DIM, grid_size=4)
    ctx = make_ctx()
    acc = probe.token_accuracies(ctx.patch, make_batch()["domain"])
    assert acc.shape == (16,)
    assert ((0 <= acc) & (acc <= 1)).all()
    # losses 里多了 patch_tokens 项
    assert "patch_tokens" in probe.losses(ctx, make_batch()["domain"])


def test_token_probe_does_not_backprop_into_encoder():
    """逐 token 探针同样是仪器：反传后 encoder 特征不得有梯度。"""
    probe = DomainProbe(dim=DIM, grid_size=4)
    ctx = make_ctx(requires_grad=True)
    probe.token_loss(ctx.patch, make_batch()["domain"]).backward()
    assert ctx.patch.grad is None
    assert probe.tok_w.grad is not None


def test_token_probe_requires_grid_size():
    probe = DomainProbe(dim=DIM)          # 未启用逐 token
    with pytest.raises(RuntimeError, match="grid_size"):
        probe.token_loss(make_ctx().patch, make_batch()["domain"])
