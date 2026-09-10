"""损失函数模块（guideline v2 §5.2 / §6）。

- grl:        梯度反转层（域对抗）
- depth:      L_metric（异方差 Laplacian NLL）、L_teacher（scale-and-shift
              对齐的教师蒸馏）、L_grad（多尺度梯度）、L_smooth（RGB 边缘感知
              平滑，仅空洞区），及组合 DepthLoss
- dyn:        隐空间动力学预测损失（smooth-L1 + 动态区域加权）

所有深度损失在 disparity（逆深度）空间、解码器输出分辨率（64×64）上计算。
"""

from .grl import GradientReversal, grad_reverse
from .depth import DepthLoss, edge_smooth, metric_nll, multiscale_grad, ssi_teacher
from .dyn import latent_prediction_loss

__all__ = [
    "GradientReversal", "grad_reverse",
    "DepthLoss", "metric_nll", "ssi_teacher", "multiscale_grad", "edge_smooth",
    "latent_prediction_loss",
]
