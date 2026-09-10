"""深度解码器 D：仅接 z 的轻量卷积上采样头，双头输出 (μ, log σ)。

规格（设计文档 §8，guideline v2 §2.3）：
- 输入只允许 z（256×384，**禁止 encoder skip connection**——防 RGB 捷径，
  这是"几何在隐空间"论证的根基，代码评审检查项）；
- 结构：reshape 16×16×384 →（可选拼接归一化 uv 坐标 2 通道）→ 1×1 conv
  384→256 → 上采样 16→32（2×[conv3×3+GroupNorm+GELU]，256ch）→
  上采样 32→64（同结构，128ch）→ 3×3 conv → 2 通道；
- 输出 μ（disparity，无激活）与 log σ（clamp 到 [-3, 5] 防塌缩/爆炸）；
- GroupNorm 与 batch 大小无关，小 batch 稳定；
- 欠拟合时只准加宽通道，**不得**接 skip 或加注意力（§8 末条）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(c_in: int, c_out: int, n_conv: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n_conv):
        layers += [
            nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.GELU(),
        ]
    return nn.Sequential(*layers)


class DepthDecoder(nn.Module):
    def __init__(self, d: int = 384, grid_size: int = 16, out_size: int = 64,
                 width1: int = 256, width2: int = 128, use_uv: bool = True,
                 log_sigma_min: float = -3.0, log_sigma_max: float = 5.0) -> None:
        super().__init__()
        if out_size != grid_size * 4:
            raise ValueError(
                f"当前结构为两级 2× 上采样，要求 out_size = 4×grid_size；"
                f"grid={grid_size} 时 out_size 应为 {grid_size * 4}，实际 {out_size}")
        self.grid_size = grid_size
        self.out_size = out_size
        self.use_uv = use_uv
        self.log_sigma_min, self.log_sigma_max = log_sigma_min, log_sigma_max

        c_in = d + 2 if use_uv else d
        self.in_proj = nn.Conv2d(c_in, width1, 1)
        self.up1 = _conv_block(width1, width1)   # 16→32，256ch
        self.up2 = _conv_block(width1, width2)   # 32→64，128ch
        self.head = nn.Conv2d(width2, 2, 3, padding=1)

        if use_uv:
            g = grid_size
            uv = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, g), torch.linspace(-1, 1, g),
                indexing="xy"))                   # (2, g, g)
            self.register_buffer("uv", uv[None], persistent=False)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (B, grid², d) → (μ, log σ)，各 (B, out_size, out_size)。"""
        B, N, D = z.shape
        g = self.grid_size
        if N != g * g:
            raise ValueError(f"token 数 {N} != grid²={g * g}")
        x = z.transpose(1, 2).reshape(B, D, g, g)
        if self.use_uv:
            x = torch.cat([x, self.uv.expand(B, -1, -1, -1).to(x.dtype)], dim=1)
        x = self.in_proj(x)                                   # (B, w1, 16, 16)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)                                       # (B, w1, 32, 32)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)                                       # (B, w2, 64, 64)
        out = self.head(x)                                    # (B, 2, 64, 64)
        mu = out[:, 0]
        log_sigma = out[:, 1].clamp(self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma
