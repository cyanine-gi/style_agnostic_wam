"""隐空间动力学预测损失（guideline v2 §6 Stage 1，设计文档 §7）。

    L_dyn = smooth_l1( ᑮ_{t+1}, sg(z_{t+1}) )  + 多步展开版（unroll k∈{1,2,4}）

- per-patch smooth-L1，目标一律 stop-gradient（调用侧传 detach 后的目标，
  本函数内部再 detach 一次以作保险）；
- 动态区域加权：权重 ∝ 相邻帧 latent 差（默认 ‖z_target − z_prev‖，
  即 z_{t+1} 与 z_t 之差；也可由调用侧传 z_t 与 z_{t−1}），逐样本归一到
  均值 1，防止容量浪费在静止背景；
- 多步展开在训练脚本里循环调用 TransitionModel.forward 后对每步求本损失。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_prediction_loss(z_hat: torch.Tensor, z_target: torch.Tensor,
                           z_prev: torch.Tensor | None = None,
                           huber_beta: float = 1.0,
                           dyn_weight: bool = True,
                           eps: float = 1e-6) -> torch.Tensor:
    """z_hat, z_target, z_prev: (B, N, d)。返回标量损失。"""
    target = z_target.detach()
    per_patch = F.smooth_l1_loss(z_hat, target, reduction="none",
                                 beta=huber_beta).mean(dim=-1)   # (B, N)
    if dyn_weight and z_prev is not None:
        w = (target - z_prev.detach()).norm(dim=-1)              # (B, N)
        w = w / w.mean(dim=1, keepdim=True).clamp(min=eps)
        per_patch = per_patch * w
    return per_patch.mean()
