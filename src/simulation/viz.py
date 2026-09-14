"""可视化工具：与数据侧（scripts/check_franka_dataset.py）保持一致。

深度伪彩色逻辑必须和 check_franka_dataset.depth_to_rgb 相同
（2026-09-14 用户裁决：可视化逻辑对齐，避免"看起来不一样"的假象）：
每帧 5–95 分位拉伸到 [0,1]，红=远、蓝=近、无效（<=0）=黑。

深度数值语义（2026-09-14 用户裁决）：**统一 RoboMIND 约定 = z-depth
（轴向，distance_to_image_plane）、单位毫米、无效=0、1mm 整数量化
（对齐 uint16 mm 精度）**。仿真侧原始输出是 float 米，在读取边界
（read_depth_mm / mdp.camera_depth）×1000 并四舍五入到整数 mm。
剩余固有差异只剩洞率（RoboMIND 有洞，仿真稠密无洞）。
"""

from __future__ import annotations

import numpy as np


def read_depth_mm(cam) -> np.ndarray:
    """从 IsaacLab Camera 读 env0 单帧深度，返回 (H,W) float32 整数 mm
    （无效=0；取值为整数，量化粒度对齐 RoboMIND uint16 mm）。

    取 distance_to_image_plane（z-depth，轴向），原始为米，×1000。
    """
    import torch
    d = cam.data.output["distance_to_image_plane"][0, ..., 0]
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d = torch.round(d * 1000.0).clamp_(min=0.0, max=65535.0)
    return d.cpu().numpy().astype(np.float32)


def depth_to_rgb(d: np.ndarray) -> np.ndarray:
    """深度（任意单位，>0 有效）→ 伪彩色 float RGB ∈ [0,1]，形状 (H,W,3)。"""
    d = np.asarray(d, dtype=np.float32)
    valid = d > 0
    out = np.zeros((*d.shape, 3), dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(d[valid], [5, 95])
        t = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        out[..., 0] = t
        out[..., 2] = 1 - t
        out[..., 1] = 0.5 * (1 - np.abs(t - 0.5) * 2)
        out[~valid] = 0
    return out

