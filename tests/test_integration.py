"""端到端冒烟测试：E → (ActionAdapter → T) → D + 判别器/GRL + 全部损失一次反传。

验证各模块接口互相咬合（guideline §2 架构图），用真实编码器权重；权重缺失时跳过。
T 走 teacher-forced block-causal 多步前向（已裁决 2026-09-11 的逐步 token 方案）。
"""

import pytest
import torch

from conftest import ENCODER_PATH

pytestmark = pytest.mark.skipif(not ENCODER_PATH.is_dir(),
                                reason="编码器权重未下载")

from sawvla.losses import DepthLoss, grad_reverse, latent_prediction_loss  # noqa: E402
from sawvla.models import (ActionAdapter, DINOv2Encoder,  # noqa: E402
                           DepthDecoder, DomainDiscriminator, TransitionModel)


def test_full_chain_one_step():
    torch.manual_seed(0)
    E = DINOv2Encoder(ENCODER_PATH)
    adapter = ActionAdapter(action_dim=14, proprio_dim=16)
    T = TransitionModel(depth=2, n_heads=6)
    with torch.no_grad():
        # head 零初始化 ⇒ 首个前向 cond/T 主干梯度恰为 0（恒等起步的设计行为，
        # 真实训练第一步 optimizer.step 后即解除）；测试里先打破零初始化
        T.head.weight.normal_(0, 0.02)
    D = DepthDecoder()
    disc = DomainDiscriminator()
    crit = DepthLoss()

    k = 2                                        # 多步：帧 0,1 → 预测帧 1,2
    rgb = torch.randn(1, k + 1, 3, 224, 224)
    z = [E(rgb[:, i]) for i in range(k + 1)]
    z_seq = torch.stack(z[:k], dim=1)            # (1, k, 256, 384)

    act_tok, prop_tok = adapter(torch.randn(1, k, 14), torch.randn(1, k, 16))
    z_hat = T(z_seq, act_tok, prop_tok)          # (1, k, 256, 384)

    # Stage 0：深度损失反传进 E；Stage 1 起调用侧对 z 做 detach（本测试不分离，
    # 只验证通路）
    mu, log_sigma = D(z_hat[:, 0])
    d_target = torch.rand(1, 64, 64)
    mask = (torch.rand(1, 64, 64) > 0.2).float()
    teacher = torch.rand(1, 64, 64)
    losses = crit(mu, log_sigma, d_target, mask, teacher, rgb[:, 1])

    l_dyn = latent_prediction_loss(z_hat[:, 0], z[1], z_prev=z[0])

    # GRL：判别器同时作用 z_t 与 ᑮ，λ ramp 期间的某个中间值
    logit_z = disc(grad_reverse(z[0], lambd=0.05))
    logit_p = disc(grad_reverse(z_hat[:, 0], lambd=0.05))
    l_adv = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.cat([logit_z, logit_p]), torch.tensor([1.0, 0.0]))

    total = losses["total"] + l_dyn + l_adv
    total.backward()

    assert torch.isfinite(total)
    for name, module in [("E", E), ("adapter", adapter), ("T", T),
                         ("D", D), ("disc", disc)]:
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), \
            f"{name} 没有收到梯度"
