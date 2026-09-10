"""梯度反转层（GRL）：前向恒等，反向把梯度乘以 -λ。

用法（guideline v2 §2.2、Stage 2）：
    logit_z = discriminator(grad_reverse(z_t, lambd))       # 反转进 E
    logit_p = discriminator(grad_reverse(z_hat, lambd))     # 反转进 T
    L_adv = BCE(logit_z, domain) + BCE(logit_p, domain)
E/T 的优化目标是**增大**判别器损失（消灭域信息），判别器自身用同一 BCE
正常更新（不经过 GRL 的前向再算一次，或用 detach 特征）。

λ 必须从 0 缓慢 ramp 到 0.05–0.1（~5k step），禁止一步加满（§6 Stage 2）。
"""

from __future__ import annotations

import torch


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)
