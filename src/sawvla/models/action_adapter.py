"""动作适配器：动作 + 本体感 → 逐步条件 token，只进转移模型 T，不进编码器 E。

规格（已裁决 2026-09-11，overall_tensor_flow.md §1.2，guideline v2 §2.3）：
- **逐步 token 化**：每步动作向量独立 MLP 升为 1 个 action token，k 步 = k 个
  token，禁止整块压缩（整块压缩会让第 1 步预测看到未来动作，与 block-causal
  因果语义矛盾）；
- 本体感向量（全局量）每步 1 个 proprio token；
- k 可变（帧跳/展开步数由采样决定），本模块对步数无假设；
- 这些 token 不携带空间位置编码（在 T 内部只加时间维编码，见 transition.py）。

action_dim / proprio_dim 由 config 传入（`[已核实 2026-09-11]` Franka 数据
= 16 维：双臂 7 关节 + 双夹爪，见 configs/data.yaml curves 段与
sawvla.data.FrankaJointGripperPreprocessor），本模块不硬编码默认值。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionAdapter(nn.Module):
    def __init__(self, action_dim: int, proprio_dim: int, d: int = 384,
                 hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or 2 * d
        self.d = d
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, actions: torch.Tensor,
                proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """actions: (B, k, action_dim)；proprio: (B, k, proprio_dim)。

        返回 (act_tokens, proprio_tokens)，各 (B, k, d)。第 j 个 action token
        只含第 j 步动作——T 的 block-causal 掩码保证第 j 步预测只见 a_{≤j}。
        """
        if actions.ndim != 3 or proprio.ndim != 3 or actions.shape[:2] != proprio.shape[:2]:
            raise ValueError(
                f"actions 应为 (B, k, action_dim)、proprio (B, k, proprio_dim)，"
                f"实际 {tuple(actions.shape)} / {tuple(proprio.shape)}")
        return self.action_mlp(actions), self.proprio_mlp(proprio)
