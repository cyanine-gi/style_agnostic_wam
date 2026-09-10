"""域判别器 + GRL 单测：形状、容量上限、梯度符号确实反转（§11.3-3）。"""

import torch

from sawvla.losses import grad_reverse
from sawvla.models import DomainDiscriminator


def test_output_shape():
    disc = DomainDiscriminator()
    logit = disc(torch.randn(4, 256, 384))
    assert logit.shape == (4,)


def test_small_capacity():
    disc = DomainDiscriminator()
    n = sum(p.numel() for p in disc.parameters()) / 1e6
    assert n < 2.0                             # 刻意小容量（~1M），防 GRL 不稳


def test_grl_negates_gradient():
    # 固定判别器，验证 E 侧梯度方向反转：经 GRL 的梯度 = -λ × 不经 GRL 的梯度
    torch.manual_seed(0)
    disc = DomainDiscriminator()
    z = torch.randn(2, 256, 384, requires_grad=True)
    logit_plain = disc(z)
    g_plain = torch.autograd.grad(logit_plain.sum(), z, retain_graph=True)[0]
    logit_grl = disc(grad_reverse(z, lambd=0.5))
    g_grl = torch.autograd.grad(logit_grl.sum(), z)[0]
    assert torch.allclose(g_grl, -0.5 * g_plain, atol=1e-6)


def test_grl_lambda_ramp_semantics():
    # λ=0 时上游完全收不到对抗梯度（ramp 起点的正确行为）
    disc = DomainDiscriminator()
    z = torch.randn(2, 256, 384, requires_grad=True)
    disc(grad_reverse(z, lambd=0.0)).sum().backward()
    assert z.grad.abs().sum() == 0.0


def test_grl_forward_is_identity():
    z = torch.randn(2, 256, 384)
    assert torch.equal(grad_reverse(z, 1.0), z)
