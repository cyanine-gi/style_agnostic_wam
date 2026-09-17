"""HSIC 独立性惩罚（2026-09-17 裁决·三，见 src/sawvla/train_failed_log.md）。

非对抗清洗目标：直接惩罚特征与域标签的统计依赖，无判别器、无对弈动力学
（run1 追不上 / run2 饱和死 / run3 overshoot 振荡三连败后的路线切换）。
二值域标签下 HSIC ≈ 两域特征分布的 MMD。

用法（Stage 2，双挂点：E 侧全帧 reg_agnostic + T 侧每步 r̂）：
    loss = normalized_hsic(torch.cat([feat_cur, buf_x]), torch.cat([y_cur, buf_y]))
当前 batch 样本带梯度回传 E/T，跨步 FIFO 缓冲样本 detach 只作统计基底
（单 batch 48/32 样本估计噪声太大，缓冲是硬性要求）。
"""

from __future__ import annotations

import torch


def normalized_hsic(x: torch.Tensor, y: torch.Tensor,
                    sigma: torch.Tensor | None = None) -> torch.Tensor:
    """归一化 HSIC（有偏估计，中心化核 trace(KHLH)/n²）。

    x: (n, ...) 连续特征（内部 flatten，RBF 核，带宽 = 成对距离中位数）；
    y: (n,) 二值域标签（相等核）。返回标量 ∈ [0,1] 量级，0 = 独立。
    """
    n = x.shape[0]
    x = x.flatten(1).float()
    y = y.flatten().float().to(x.device)
    dist = torch.cdist(x, x)
    if sigma is None:
        sigma = dist.detach().median().clamp_min(1e-6)
    K = torch.exp(-dist.pow(2) / (2 * sigma * sigma))
    L = (y[:, None] == y[None, :]).to(x.dtype)
    H = (torch.eye(n, device=x.device, dtype=x.dtype) - 1.0 / n)
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic_xy = (Kc * Lc).sum() / (n * n)
    hsic_xx = (Kc * Kc).sum() / (n * n)
    hsic_yy = (Lc * Lc).sum() / (n * n)
    return hsic_xy / (hsic_xx * hsic_yy).clamp_min(1e-12).sqrt()


class HSICBuffer:
    """跨步 FIFO 特征缓冲（detach 统计基底；梯度只经当前 batch 回传）。"""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None

    def push(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x, y = x.detach(), y.detach()
        if self.x is None:
            self.x, self.y = x, y
        else:
            self.x = torch.cat([self.x, x])
            self.y = torch.cat([self.y, y])
        if self.x.shape[0] > self.capacity:
            self.x = self.x[-self.capacity:]
            self.y = self.y[-self.capacity:]

    def join(self, x: torch.Tensor, y: torch.Tensor):
        """当前 batch（带梯度）拼上缓冲（detach）。"""
        if self.x is None:
            return x, y
        return torch.cat([x, self.x]), torch.cat([y, self.y])
