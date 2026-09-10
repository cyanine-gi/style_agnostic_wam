"""视觉编码器 E：DINOv2-S/14（registers 版）初始化，全程可微调，禁止冻结。

规格（guideline v2 §2.2/§2.3）：
- 输入 224×224 RGB（ImageNet 归一化由 dataloader 负责）；
- 输出 16×16=256 个 patch token，d=384，2D 网格结构保留，禁止全局池化；
- CLS 与 4 个 register token 不进隐空间（隐空间必须是纯空间网格，便于 reshape）；
- 初始化后全程可微调——GRL 除域与 Stage 0 深度塑形都需要梯度进入 E；
- 防漂移锚（冻结副本 + 特征蒸馏正则）在训练脚本侧构建：再实例化一份并
  requires_grad_(False) 即可，本模块不内置。

权重必须来自本地目录（configs/model.yaml: paths.encoder），禁止联网拉取。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv2Encoder(nn.Module):
    """DINOv2 patch token 编码器。

    Args:
        model_path: 本地 HF 格式权重目录（dinov2-with-registers-small）。
        grid_size:  输出空间网格边长（224/14=16）。
        image_size: 输入边长，默认 224；与权重预训练分辨率（518）不同时
                    自动插值 2D 位置编码（DINOv2 原生支持，uv 绑定不破坏）。
    """

    def __init__(self, model_path: str | Path, grid_size: int = 16,
                 image_size: int = 224) -> None:
        super().__init__()
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"编码器权重目录不存在: {model_path}（先运行权重下载脚本）")
        # AutoModel 必须解析到 Dinov2WithRegistersModel；若解析成 Dinov2Model
        # 说明 transformers 版本过旧或权重不对（register token 会被静默丢弃）。
        self.backbone = AutoModel.from_pretrained(str(model_path))
        n_reg = int(getattr(self.backbone.config, "num_register_tokens", 0))
        if type(self.backbone).__name__ != "Dinov2WithRegistersModel" or n_reg <= 0:
            raise RuntimeError(
                f"期望 Dinov2WithRegistersModel（registers>0），实际 "
                f"{type(self.backbone).__name__} num_register_tokens={n_reg}")

        self.grid_size = grid_size
        self.image_size = image_size
        self.num_registers = n_reg
        self.dim = int(self.backbone.config.hidden_size)
        self.num_patch_tokens = grid_size * grid_size

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: (B, 3, H, W) 已归一化 → z: (B, grid_size², d) 连续 latent。"""
        out = self.backbone(pixel_values=rgb, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state  # (B, 1 + n_reg + N, d)
        patch = tokens[:, 1 + self.num_registers:, :]
        if patch.shape[1] != self.num_patch_tokens:
            raise ValueError(
                f"patch token 数 {patch.shape[1]} != grid_size²={self.num_patch_tokens}，"
                f"输入分辨率 {tuple(rgb.shape[-2:])} 与 image_size={self.image_size} 不一致")
        return patch

    def freeze(self) -> None:
        """Stage 1 冻结 E 用（或构建防漂移锚副本后调用）。"""
        self.requires_grad_(False)

    def unfreeze(self) -> None:
        self.requires_grad_(True)
