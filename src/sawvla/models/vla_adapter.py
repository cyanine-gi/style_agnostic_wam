"""VLA 侧 latent 注入 adapter：WAM 的 256 个 patch token → 64 个 LLM 软 token。

**定位：本模块属于 VLA 侧，跟着 VLA 走，不是 WAM 本体的一部分**
（已裁决 2026-09-11：WAM 模块不感知 VLA 内部实现，对外接口恒定为
256×384 patch token；是否压缩、如何压缩是本模块的内部事务）。

结构（按用户指定方案）：
    z (B,256,384) → reshape 16×16
    → per-token Linear(d→d_inner) + GELU        # ≈ 1×1 降维卷积，逐 token 独立
    → space-to-depth 2×2                         # ≈ 2×2 降大小恒等卷积：无损重排
    → (B, 8×8=64, 4·d_inner)                     # 默认 4×512 = 2048 = Qwen3-VL-2B hidden
    → [可选 Linear 到 llm_dim，默认恰好相等时省略] → LayerNorm

性质与注意事项：
- 绝对几何信息无损：space-to-depth 是纯重排；逐 token Linear 不跨空间混合；
- 输出 8×8 网格与 Qwen3-VL 原生视觉 token（224 图经 patch14+2×2 merger）
  同构，手工 MRoPE (h,w) 可 1:1 对齐（overall_tensor_flow.md §2.2/§2.3）；
- **顺序约定**：每个输出 token 的 4·d_inner 通道按 2×2 块内 row-major
  [(0,0),(0,1),(1,0),(1,1)] 拼接，MRoPE position_ids 生成必须遵守同一约定；
- d_inner 的选择刻意凑 4·d_inner == llm_dim；换 VLM 或 WAM 升维 d 时必须
  同步改 config（configs/model.yaml: vla_adapter）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def space_to_depth(x: torch.Tensor, s: int = 2) -> torch.Tensor:
    """(B, H, W, C) → (B, H/s, W/s, s²·C)，块内 row-major 拼接。"""
    B, H, W, C = x.shape
    if H % s or W % s:
        raise ValueError(f"H={H}, W={W} 必须整除 s={s}")
    x = x.reshape(B, H // s, s, W // s, s, C)
    x = x.permute(0, 1, 3, 2, 4, 5)          # (B, H/s, W/s, s, s, C)
    return x.reshape(B, H // s, W // s, s * s * C)


class VLALatentAdapter(nn.Module):
    def __init__(self, d: int = 384, d_inner: int = 512, llm_dim: int = 2048,
                 grid_size: int = 16, pool: int = 2) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.pool = pool
        self.d_inner = d_inner
        merged = pool * pool * d_inner

        self.in_proj = nn.Sequential(nn.Linear(d, d_inner), nn.GELU())
        self.out_proj = (nn.Identity() if merged == llm_dim
                         else nn.Linear(merged, llm_dim))
        self.out_norm = nn.LayerNorm(llm_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, grid², d) → (B, (grid/pool)², llm_dim)。"""
        B, N, D = z.shape
        g = self.grid_size
        if N != g * g:
            raise ValueError(f"token 数 {N} != grid²={g * g}")
        x = self.in_proj(z)                          # (B, 256, d_inner)
        x = x.reshape(B, g, g, self.d_inner)
        x = space_to_depth(x, self.pool)             # (B, 8, 8, 4·d_inner)
        x = x.reshape(B, -1, self.pool * self.pool * self.d_inner)
        return self.out_norm(self.out_proj(x))       # (B, 64, llm_dim)
