"""关节位置回归信号：双臂 14 维（左 7 + 右 7），剔除夹爪。

剔除夹爪的裁决（2026-09-14）：sim 夹爪是连续行程 [0, 0.04] m，real EE
是二值开合语义，跨域不同纲、不可比；臂关节两边同为绝对关节角（rad），
跨域同纲——所以这个信号天然是一个弱域对齐锚。

读 reg_agnostic（分区纪律）：本体感是帧级全局量，属于全局槽位；
同时它跨域同纲，不会把域信息引进域无关槽位。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import SignalHead, register_signal


@register_signal("joint_pos")
class JointPosSignal(SignalHead):
    """reg_agnostic (B, 2, d) → MLP → (B, 14) 关节角回归，MSE。"""

    reads = "reg_agnostic"
    # proprio 契约 [L7, R7, gL, gR]（configs/data.yaml）：取臂、剔夹爪
    ARM_IDX = list(range(0, 7)) + list(range(7, 14))

    def __init__(self, dim: int, n_tokens: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(n_tokens * dim, 256), nn.GELU(),
            nn.Linear(256, len(self.ARM_IDX)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                                   # (B, 14)

    def target(self, batch: dict) -> torch.Tensor:
        return batch["proprio"][:, self.ARM_IDX]             # (B, 14)

    def loss(self, pred: torch.Tensor,
             target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(pred, target)
