"""深度解码器 D 单测：输出规格、log σ clamp、梯度只从 z 来。"""

import torch

from sawvla.models import DepthDecoder


def test_output_spec():
    dec = DepthDecoder()
    z = torch.randn(2, 256, 384)
    mu, log_sigma = dec(z)
    assert mu.shape == (2, 64, 64)
    assert log_sigma.shape == (2, 64, 64)


def test_log_sigma_clamped():
    dec = DepthDecoder(log_sigma_min=-3.0, log_sigma_max=5.0)
    # 人为放大 head 偏置，验证 clamp 生效（防 σ 塌缩/爆炸）
    with torch.no_grad():
        dec.head.bias[1] = 100.0
    _, log_sigma = dec(torch.randn(1, 256, 384))
    assert log_sigma.max() <= 5.0 and log_sigma.min() >= -3.0


def test_uv_channels_used():
    dec = DepthDecoder(use_uv=True)
    assert dec.in_proj.in_channels == 384 + 2
    dec_nouv = DepthDecoder(use_uv=False)
    assert dec_nouv.in_proj.in_channels == 384


def test_gradient_flows_to_z():
    dec = DepthDecoder()
    z = torch.randn(1, 256, 384, requires_grad=True)
    mu, log_sigma = dec(z)
    (mu.sum() + log_sigma.sum()).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_param_count():
    dec = DepthDecoder()
    n = sum(p.numel() for p in dec.parameters()) / 1e6
    # 规格结构（256/128 通道两级上采样）实际 ≈1.7M；轻量是特性（探针语义），
    # 此处只防结构被意外改重。
    assert 1.0 < n < 15


def test_rejects_wrong_token_count():
    dec = DepthDecoder()
    try:
        dec(torch.randn(1, 100, 384))
        raise AssertionError("应拒绝非 grid² 的 token 数")
    except ValueError:
        pass
