"""动作条件转移模型 T：block-causal 多步结构（已裁决 2026-09-11 版）。

规格（overall_tensor_flow.md §1.2/§3.2，设计文档 §7，guideline v2 §2.3）：
- **序列组织**：逐帧 block 交错——block j = [z_j 的 256 个 patch token,
  proprio_j token, a_j token]（a_j = 第 j 帧执行的动作，产生第 j+1 帧）；
- **block-causal**：block j 只能 attend block ≤ j ⇒ 第 j+1 帧预测只见
  a_{≤j}，禁止看未来动作；帧内 256 个 patch token 之间双向；
- **teacher forcing**：训练时一次前向输入真值 z_{0..k-1}，并行读出 k 步
  预测 ẑ_{1..k}（V-JEPA 2-AC 式）；推理 rollout 用 unroll() 逐步回喂 ẑ；
- patch token 用 2D-RoPE（与 E 的 16×16 网格对齐）；动作/本体感 token 不做
  空间旋转，只随所属 block 加帧索引 embedding（时间维编码）；
- 输出为**残差 delta**：ẑ_{j+1} = z_j + head(LN(x_j[:256]))，head 零初始化，
  训练初期严格恒等起步，稳定早期训练。

因果泄漏由 tests/test_transition.py::test_causal_leakage 钉死
（overall §3.4：改未来动作，过去预测必须逐位不变）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _axis_rope_tables(grid: int, half_dim: int, base: float = 10000.0):
    """生成单轴 RoPE 的 cos/sin 表。

    half_dim: 每个轴占用的通道数（head_dim 的一半）。
    返回 cos_h, sin_h, cos_w, sin_w，各 (grid², half_dim)，行间为 h 轴、列间为 w 轴。
    """
    inv_freq = base ** (-torch.arange(0, half_dim, 2).float() / half_dim)  # (half_dim//2,)
    pos = torch.arange(grid * grid).float()
    rows, cols = pos.div(grid, rounding_mode="floor"), pos % grid
    ang_h = rows[:, None] * inv_freq[None, :]   # (N, half_dim//2)
    ang_w = cols[:, None] * inv_freq[None, :]
    # rotate_half 约定：emb 复制一份拼成 half_dim
    emb_h = torch.cat([ang_h, ang_h], dim=-1)
    emb_w = torch.cat([ang_w, ang_w], dim=-1)
    return emb_h.cos(), emb_h.sin(), emb_w.cos(), emb_w.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_2d_rope(q: torch.Tensor, tables) -> torch.Tensor:
    """q: (B, n_heads, k, n_patch, head_dim)；tables: (cos_h, sin_h, cos_w, sin_w)。

    head_dim 前半做 h 轴旋转、后半做 w 轴旋转，两轴互不混叠。
    """
    cos_h, sin_h, cos_w, sin_w = tables
    hd = q.shape[-1]
    q_h, q_w = q[..., : hd // 2], q[..., hd // 2:]
    dtype = q.dtype
    q_h = q_h * cos_h.to(dtype) + _rotate_half(q_h) * sin_h.to(dtype)
    q_w = q_w * cos_w.to(dtype) + _rotate_half(q_w) * sin_w.to(dtype)
    return torch.cat([q_h, q_w], dim=-1)


class _Attention(nn.Module):
    """MHA + block-causal 掩码；每个 block 内前 n_patch 个 token 施加 2D-RoPE。"""

    def __init__(self, d: int, n_heads: int, n_patch: int, n_block: int,
                 rope_tables) -> None:
        super().__init__()
        if d % n_heads != 0 or (d // n_heads) % 2 != 0:
            raise ValueError(f"d={d} / n_heads={n_heads} 无法均分且按 2D-RoPE 二分")
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.n_patch = n_patch
        self.n_block = n_block
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        cos_h, sin_h, cos_w, sin_w = rope_tables
        # (1, 1, 1, N, head_dim//2)，fp32 常驻，forward 时转 dtype
        for name, t in [("cos_h", cos_h), ("sin_h", sin_h),
                        ("cos_w", cos_w), ("sin_w", sin_w)]:
            self.register_buffer(name, t[None, None, None], persistent=False)

    def _rope(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """x: (B, H, k*n_block, hd) → 对每个 block 的 patch 段做 2D-RoPE。"""
        B, H, L, hd = x.shape
        x = x.reshape(B, H, k, self.n_block, hd)
        xp, xc = x[..., : self.n_patch, :], x[..., self.n_patch:, :]
        xp = _apply_2d_rope(xp, (self.cos_h, self.sin_h, self.cos_w, self.sin_w))
        return torch.cat([xp, xc], dim=3).reshape(B, H, L, hd)

    def forward(self, x: torch.Tensor, k: int,
                attn_mask: torch.Tensor | None) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, kk, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)     # (B, H, L, hd)
        q, kk = self._rope(q, k), self._rope(kk, k)
        o = F.scaled_dot_product_attention(q, kk, v, attn_mask=attn_mask)
        return self.proj(o.transpose(1, 2).reshape(B, L, D))


class _Block(nn.Module):
    def __init__(self, d: int, n_heads: int, n_patch: int, n_block: int,
                 rope_tables, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = _Attention(d, n_heads, n_patch, n_block, rope_tables)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)), nn.GELU(),
            nn.Linear(int(d * mlp_ratio), d),
        )

    def forward(self, x: torch.Tensor, k: int,
                attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), k, attn_mask)
        return x + self.mlp(self.ln2(x))


class TransitionModel(nn.Module):
    """block-causal 多步转移模型：真值 z_{0..k-1} + 逐步条件 → ẑ_{1..k}。

    Args:
        d:            latent 通道数（与 E 一致，384）。
        depth:        Transformer 层数（默认 8，≈20M 参数）。
        n_heads:      注意力头数（默认 6）。
        grid_size:    patch 网格边长（16）。
        n_cond:       每帧条件 token 数（1 本体感 + 1 动作 = 2）。
        max_frames:   帧索引 embedding 表大小（多步展开的最大步数）。
    """

    def __init__(self, d: int = 384, depth: int = 8, n_heads: int = 6,
                 grid_size: int = 16, n_cond: int = 2,
                 max_frames: int = 16) -> None:
        super().__init__()
        self.d = d
        self.grid_size = grid_size
        self.n_patch = grid_size * grid_size
        self.n_cond = n_cond
        self.n_block = self.n_patch + n_cond

        rope_tables = _axis_rope_tables(grid_size, (d // n_heads) // 2)
        self.blocks = nn.ModuleList(
            _Block(d, n_heads, self.n_patch, self.n_block, rope_tables)
            for _ in range(depth))
        self.frame_embed = nn.Embedding(max_frames, d)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, d)
        # 恒等起步：ẑ = z + head(...)，head 零初始化 ⇒ 初始 ẑ ≡ z
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _causal_mask(self, k: int, device) -> torch.Tensor:
        """(k*n_block, k*n_block) bool；True = 允许 attend（block j 见 ≤j）。"""
        block_of = torch.arange(k * self.n_block, device=device) // self.n_block
        return block_of[None, :] <= block_of[:, None]

    def forward(self, z_seq: torch.Tensor, act_tokens: torch.Tensor,
                proprio_tokens: torch.Tensor,
                frame_offset: int = 0) -> torch.Tensor:
        """teacher-forced 多步前向。

        z_seq:          (B, k, grid², d)，真值 latent z_{0..k-1}；
        act_tokens:     (B, k, d)，ActionAdapter 输出，a_j 作用于第 j 帧；
        proprio_tokens: (B, k, d)；
        frame_offset:   首个 block 的帧索引（unroll 时传当前步号）。

        返回 ẑ: (B, k, grid², d)，第 j 项是第 j+1 帧的预测。
        """
        B, k, N, D = z_seq.shape
        if N != self.n_patch:
            raise ValueError(f"patch token 数 {N} != grid²={self.n_patch}")
        if act_tokens.shape[:2] != (B, k) or proprio_tokens.shape[:2] != (B, k):
            raise ValueError(
                f"条件 token 步数与 z_seq 不一致: z (B,{k},...) vs "
                f"act {tuple(act_tokens.shape)} / proprio {tuple(proprio_tokens.shape)}")

        # block j = [z_j (256), proprio_j, a_j]
        x = torch.cat([z_seq,
                       proprio_tokens[:, :, None],
                       act_tokens[:, :, None]], dim=2)      # (B, k, n_block, d)
        fr = torch.arange(frame_offset, frame_offset + k, device=x.device)
        x = x + self.frame_embed(fr)[None, :, None, :]
        x = x.reshape(B, k * self.n_block, D)

        attn_mask = self._causal_mask(k, x.device) if k > 1 else None
        for blk in self.blocks:
            x = blk(x, k, attn_mask)

        x = x.reshape(B, k, self.n_block, D)[:, :, : self.n_patch]
        return z_seq + self.head(self.ln_f(x))

    @torch.no_grad()
    def unroll(self, z0: torch.Tensor, act_tokens: torch.Tensor,
               proprio_tokens: torch.Tensor) -> torch.Tensor:
        """推理 rollout：逐步把 ẑ 回喂为下一步输入，不再依赖 RGB。

        act_tokens / proprio_tokens: (B, k, d)。返回 (B, k, grid², d)。
        训练用多步损失直接调 forward（teacher forcing）；若需自由展开训练
        （feed-back），在训练脚本里循环调用 forward 并保留梯度，勿用本函数。
        """
        k = act_tokens.shape[1]
        preds, z = [], z0
        for j in range(k):
            z = self.forward(z[:, None], act_tokens[:, j:j + 1],
                             proprio_tokens[:, j:j + 1], frame_offset=j)[:, 0]
            preds.append(z)
        return torch.stack(preds, dim=1)
