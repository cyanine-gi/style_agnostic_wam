"""域探针：三通道域分类准确率监控（独立组件，不走信号注册表）。

2026-09-14 用户裁决（原话动机）："我们很难做到完全不含信息，但是可以
做到尽量少含信息，并且知道自己大概泄露了多少域信息，对下游用户的信心
也有显著提升。"——探针的职责是**量**，不是**治**（治是 Stage 2 GRL）。

三通道各一个线性头（patch 先 mean-pool），特征 detach——探针的梯度
绝不进 E（探针是仪器，不是损失）：
- reg_domain（域槽位）：准确率应高——锚定教师被动携带域信息；
- reg_agnostic（域无关槽位）：目标是贴近随机（50%），Stage 2 GRL 只挂这里；
- patch（局部通道）：泄漏监控。

读数解释：与 50%（二分类随机）的差 = 该通道域信息含量的运营指标。

逐 token 热图探针（2026-09-15 用户裁决）：patch 通道 256 个位置各一个
独立线性头（不注意力路由、不共享——逐位置测量"哪里在漏"），产出
16×16 泄漏热图。与前两者的分工：mean-pool 线性 = 平凡可读性（总量），
逐 token = 空间分布（哪里）。已知局限（讨论已确认）：① 逐 token 各自
55% 聚合后仍可 100% 可分（冗余累积），不替代汇总读数；② 测不到关系型
泄漏（机位差异编码在 token 间相对配置里），那维由 register 通道兜住。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DomainProbe(nn.Module):
    CHANNELS = ("patch", "reg_domain", "reg_agnostic")

    def __init__(self, dim: int = 384, n_classes: int = 2,
                 grid_size: int | None = None) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {c: nn.Linear(dim, n_classes) for c in self.CHANNELS})
        # 逐 token 探针：grid_size² 个位置各一对 (dim→n_classes) 权重
        self.grid_size = grid_size
        if grid_size is not None:
            n_tok = grid_size ** 2
            self.tok_w = nn.Parameter(
                torch.randn(n_tok, dim, n_classes) * 0.02)
            self.tok_b = nn.Parameter(torch.zeros(n_tok, n_classes))

    def _feat(self, ctx, ch: str) -> torch.Tensor:
        x = ctx.channel(ch)
        if x.dim() == 3:
            x = x.mean(dim=1)              # token 维 mean-pool
        return x.detach().float()          # 关键：仪器不反传进 E

    def losses(self, ctx, domain: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {c: torch.nn.functional.cross_entropy(
                   self.heads[c](self._feat(ctx, c)), domain)
               for c in self.CHANNELS}
        if self.grid_size is not None:
            out["patch_tokens"] = self.token_loss(ctx.patch, domain)
        return out

    @torch.no_grad()
    def accuracies(self, ctx, domain: torch.Tensor) -> dict[str, float]:
        return {c: (self.heads[c](self._feat(ctx, c)).argmax(-1) == domain
                    ).float().mean().item()
                for c in self.CHANNELS}

    # ------------------------------------------------------------------ #
    # 逐 token（热图）探针：只作用在 patch 通道
    # ------------------------------------------------------------------ #

    def _check_tok(self) -> None:
        if self.grid_size is None:
            raise RuntimeError("构造时未给 grid_size，逐 token 探针未启用")

    def token_logits(self, patch: torch.Tensor) -> torch.Tensor:
        """patch (B,N,d) → (B,N,2)；特征 detach（仪器不反传进 E）。"""
        self._check_tok()
        x = patch.detach().float()
        return torch.einsum("bnd,ndk->bnk", x, self.tok_w) + self.tok_b

    def token_loss(self, patch: torch.Tensor,
                   domain: torch.Tensor) -> torch.Tensor:
        logits = self.token_logits(patch)                  # (B,N,2)
        tgt = domain[:, None].expand(-1, logits.shape[1]).flatten()
        return torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), tgt)

    @torch.no_grad()
    def token_accuracies(self, patch: torch.Tensor,
                         domain: torch.Tensor) -> torch.Tensor:
        """逐位置域分类准确率 → (N,)，reshape (grid,grid) 即泄漏热图。"""
        logits = self.token_logits(patch)
        correct = logits.argmax(-1) == domain[:, None]     # (B,N)
        return correct.float().mean(dim=0)
