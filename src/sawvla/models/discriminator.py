"""域判别器：刻意小容量（~1M 参数），防止过强导致 GRL 训练不稳。

规格（guideline v2 §2.3）：
- patch token → 1×1 conv 降维 → 全局池化 + 3 层 MLP → 二分类 logit（real/sim）；
- 同时挂在 z_t（E 的输出）与 ᑮ_{t+1}（T 的输出）上（guideline §2.2）；
- 梯度反转由 losses.grl.grad_reverse 在调用侧施加，本模块是普通判别器；
- GRL batch 必须 50/50 域均衡（采样侧职责，见 guideline §9.1）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(self, d: int = 384, grid_size: int = 16, hidden: int = 128) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.reduce = nn.Sequential(
            nn.Conv2d(d, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, grid², d) → logit (B,)；>0 判 real（标签约定由训练侧固定）。"""
        B, N, D = z.shape
        g = self.grid_size
        if N != g * g:
            raise ValueError(f"token 数 {N} != grid²={g * g}")
        x = z.transpose(1, 2).reshape(B, D, g, g)
        x = self.reduce(x).mean(dim=(2, 3))   # 全局平均池化 → (B, hidden)
        return self.mlp(x).squeeze(-1)
