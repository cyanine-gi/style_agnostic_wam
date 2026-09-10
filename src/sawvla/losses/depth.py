"""深度监督损失（guideline v2 §5.2，设计文档 §6）。

    L_depth = λ1·L_metric + λ2·L_teacher + λ3·L_grad + λ4·L_smooth

    L_metric  = mean_{M=1}( |d − μ| / σ + log σ )   # 异方差 Laplacian NLL，仅有效点
    L_teacher = mean_all( |align(t) − μ| )          # 教师稠密蒸馏，全图
    L_grad    = Σ_s mean_{M_s=1}( |∇μ_s − ∇d_s| )   # 多尺度梯度，4 级
    L_smooth  = mean_{M=0}( |∇μ| · exp(−|∇rgb|) )   # RGB 边缘感知平滑，仅空洞区

约定：
- 全部在 disparity（逆深度）空间、解码器输出分辨率（默认 64×64）上计算；
- mask 为 float，1=有效。mask 全空的 batch 返回 0（梯度连通，不 NaN）；
- d 在 mask=0 处的取值会被忽略（调用侧可填任意有限值，NaN 会被剔除）；
- 训练侧掩码只有"有限值/量程内"一条规则（§5.1a），评估掩码另算（§7.4）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """sum(x*mask) / max(sum(mask), 1)。mask 全空时返回带梯度通路的 0。"""
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def _sanitize(d: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask=0 处置 0（防 NaN/飞点数值经 0×NaN 污染池化结果）。"""
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(mask > 0, d, torch.zeros_like(d))


def metric_nll(mu: torch.Tensor, log_sigma: torch.Tensor,
               d: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """异方差 Laplacian NLL，仅传感器有效点（§5.1a/b）。

    mu, log_sigma, d, mask: (B, H, W)。σ 由解码器学习，对"有值但不可靠"
    的像素软降权；log σ 的 clamp 在解码器内完成，此处不再处理。
    """
    d = _sanitize(d, mask)
    r = (d - mu).abs()
    loss_px = r / log_sigma.exp() + log_sigma
    return _masked_mean(loss_px, mask)


def ssi_teacher(mu: torch.Tensor, t: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """教师稠密蒸馏（§5.1c）：per-image 最小二乘 scale-and-shift 把教师
    相对深度 align 到 μ 后做全图 L1。

    方向是 align(t)→μ（教师贴预测），度量锚定由 L_metric 负责（§5.2 要点）。
    对齐系数 (a, b) 由 detach 后的 μ 统计量解出：若让梯度穿过 (a, b)，
    μ 退化为常数时 a→0、b→μ 可使本项恒为 0（塌缩捷径）；detach 后常数解
    仍会经 b 漏出，因此第一阶段必须配 L_grad 一起用（guideline §5.2 已
    注明 L_teacher 不单独跑）。
    """
    B = mu.shape[0]
    mu_f = mu.reshape(B, -1)
    t_f = torch.nan_to_num(t.reshape(B, -1), nan=0.0)
    mu_sg = mu_f.detach()

    mean_t = t_f.mean(dim=1)
    mean_m = mu_sg.mean(dim=1)
    var_t = ((t_f - mean_t[:, None]) ** 2).mean(dim=1)
    cov = ((t_f - mean_t[:, None]) * (mu_sg - mean_m[:, None])).mean(dim=1)
    a = cov / (var_t + eps)                       # (B,) detached
    b = mean_m - a * mean_t                       # (B,) detached

    aligned = a[:, None] * t_f + b[:, None]
    return (aligned - mu_f).abs().mean()


def _masked_avg_pool(d: torch.Tensor, mask: torch.Tensor, k: int,
                     thresh: float) -> tuple[torch.Tensor, torch.Tensor]:
    """掩码感知降采样：无效像素不参与池化，按有效比例重建掩码。

    d, mask: (B, 1, H, W)。返回 (d_s, m_s)，m_s = (有效比例 >= thresh)。
    """
    if k == 1:
        return d, mask
    num = F.avg_pool2d(d * mask, k)
    ratio = F.avg_pool2d(mask, k)
    d_s = num / ratio.clamp(min=_EPS)
    m_s = (ratio >= thresh).to(d.dtype)
    return d_s, m_s


def _grad_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def multiscale_grad(mu: torch.Tensor, d: torch.Tensor, mask: torch.Tensor,
                    n_scales: int = 4, pool_valid_thresh: float = 0.5) -> torch.Tensor:
    """多尺度梯度损失（Eigen et al.；§5.2 L_grad，设计文档 §6.3）。

    64→32→16→8 四级（n_scales=4，含原分辨率）。目标侧掩码感知降采样；
    梯度像素的有效性要求两端邻点均有效。掩码全空的层级贡献为 0。
    """
    d = _sanitize(d, mask)
    mu_, d_, m_ = mu[:, None], d[:, None], mask[:, None]
    total = mu.new_zeros(())
    for s in range(n_scales):
        k = 2 ** s
        d_s, m_s = _masked_avg_pool(d_, m_, k, pool_valid_thresh)
        mu_s = mu_ if k == 1 else F.avg_pool2d(mu_, k)
        gx_mu, gy_mu = _grad_xy(mu_s)
        gx_d, gy_d = _grad_xy(d_s)
        mx = m_s[..., :, 1:] * m_s[..., :, :-1]
        my = m_s[..., 1:, :] * m_s[..., :-1, :]
        total = total + _masked_mean((gx_mu - gx_d).abs(), mx) \
            + _masked_mean((gy_mu - gy_d).abs(), my)
    return total / n_scales


def edge_smooth(mu: torch.Tensor, rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """RGB 边缘感知平滑，仅空洞区（M=0）生效（§5.2 L_smooth）。

    rgb: (B, 3, H, W)，若分辨率与 mu 不同则双线性降到 mu 的尺寸——这是
    唯一允许 RGB 进入深度监督的位置，不经过 E，无域泄漏通道。
    """
    if rgb.shape[-2:] != mu.shape[-2:]:
        rgb = F.interpolate(rgb, size=mu.shape[-2:], mode="bilinear",
                            align_corners=False)
    gray = rgb.mean(dim=1)
    gx_mu, gy_mu = _grad_xy(mu)
    gx_rgb, gy_rgb = _grad_xy(gray)
    hole = 1.0 - mask
    wx = torch.exp(-gx_rgb.abs())
    wy = torch.exp(-gy_rgb.abs())
    mx = hole[..., :, 1:] * hole[..., :, :-1]
    my = hole[..., 1:, :] * hole[..., :-1, :]
    return _masked_mean(gx_mu.abs() * wx, mx) + _masked_mean(gy_mu.abs() * wy, my)


class DepthLoss(nn.Module):
    """组合深度损失 L_depth（§5.2）。默认权重 λ1=1.0, λ2=1.0, λ3=0.5, λ4=0.1。"""

    def __init__(self, lambda_metric: float = 1.0, lambda_teacher: float = 1.0,
                 lambda_grad: float = 0.5, lambda_smooth: float = 0.1,
                 n_grad_scales: int = 4, pool_valid_thresh: float = 0.5) -> None:
        super().__init__()
        self.lambda_metric = lambda_metric
        self.lambda_teacher = lambda_teacher
        self.lambda_grad = lambda_grad
        self.lambda_smooth = lambda_smooth
        self.n_grad_scales = n_grad_scales
        self.pool_valid_thresh = pool_valid_thresh

    def forward(self, mu: torch.Tensor, log_sigma: torch.Tensor,
                d: torch.Tensor, mask: torch.Tensor, teacher: torch.Tensor,
                rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """mu, log_sigma, d, mask, teacher: (B, H, W)；rgb: (B, 3, H', W')。

        返回 {"total", "metric", "teacher", "grad", "smooth"}。
        """
        parts = {
            "metric": metric_nll(mu, log_sigma, d, mask),
            "teacher": ssi_teacher(mu, teacher),
            "grad": multiscale_grad(mu, d, mask, self.n_grad_scales,
                                    self.pool_valid_thresh),
            "smooth": edge_smooth(mu, rgb, mask),
        }
        total = (self.lambda_metric * parts["metric"]
                 + self.lambda_teacher * parts["teacher"]
                 + self.lambda_grad * parts["grad"]
                 + self.lambda_smooth * parts["smooth"])
        return {"total": total, **parts}
