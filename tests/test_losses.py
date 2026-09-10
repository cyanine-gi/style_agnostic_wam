"""深度损失与动力学损失单测（§11.3-2 要求的边界情形全覆盖）。"""

import torch

from sawvla.losses import (DepthLoss, edge_smooth, latent_prediction_loss,
                           metric_nll, multiscale_grad, ssi_teacher)

B, H, W = 2, 64, 64


def _batch(mask_fill=1.0):
    torch.manual_seed(0)
    mu = torch.rand(B, H, W, requires_grad=True)
    log_sigma = torch.zeros(B, H, W, requires_grad=True)
    d = torch.rand(B, H, W)
    mask = torch.full((B, H, W), mask_fill)
    teacher = torch.rand(B, H, W)
    rgb = torch.rand(B, 3, H, W)
    return mu, log_sigma, d, mask, teacher, rgb


# ---- L_metric：异方差 Laplacian NLL ----

def test_metric_nll_empty_mask():
    mu, log_sigma, d, mask, _, _ = _batch(mask_fill=0.0)
    loss = metric_nll(mu, log_sigma, d, mask)
    assert loss.item() == 0.0
    loss.backward()                            # 空掩码必须有梯度通路且不 NaN
    assert torch.isfinite(mu.grad).all()


def test_metric_nll_all_valid_and_prefers_accurate_mu():
    mu, log_sigma, d, mask, _, _ = _batch()
    loss = metric_nll(mu, log_sigma, d, mask)
    assert torch.isfinite(loss) and loss.requires_grad
    with torch.no_grad():
        mu_accurate = d + 0.01 * torch.randn_like(d)
    assert metric_nll(mu_accurate, log_sigma.detach(), d, mask) < loss


def test_metric_nll_ignores_masked_values():
    mu, log_sigma, d, mask, _, _ = _batch()
    mask[:, :32] = 0.0
    d2 = d.clone(); d2[:, :32] = 1e6           # 掩码外改成极端值
    assert torch.allclose(metric_nll(mu, log_sigma, d, mask),
                          metric_nll(mu, log_sigma, d2, mask))


def test_metric_nll_nan_in_masked_region_safe():
    mu, log_sigma, d, mask, _, _ = _batch()
    mask[:, :32] = 0.0
    d[:, :32] = float("nan")
    assert torch.isfinite(metric_nll(mu, log_sigma, d, mask))


# ---- L_teacher：scale-and-shift 对齐 ----

def test_ssi_teacher_affine_invariant():
    mu, _, _, _, teacher, _ = _batch()
    t2 = 3.7 * teacher - 12.5                  # 教师相对深度乘加任意仿射 → 损失不变
    assert torch.allclose(ssi_teacher(mu, teacher), ssi_teacher(mu, t2),
                          atol=1e-5)


def test_ssi_teacher_zero_when_aligned():
    teacher = torch.rand(B, H, W)
    mu = (2.0 * teacher + 0.3).requires_grad_(True)   # μ 恰为教师的仿射
    assert ssi_teacher(mu, teacher).item() < 1e-5


def test_ssi_teacher_grad_flows():
    mu, _, _, _, teacher, _ = _batch()
    loss = ssi_teacher(mu, teacher)
    loss.backward()
    assert mu.grad is not None and mu.grad.abs().sum() > 0


# ---- L_grad：多尺度梯度 ----

def test_multiscale_grad_empty_mask():
    mu, _, d, mask, _, _ = _batch(mask_fill=0.0)
    loss = multiscale_grad(mu, d, mask)
    assert loss.item() == 0.0
    loss.backward()


def test_multiscale_grad_positive_on_wrong_prediction():
    mu = torch.zeros(B, H, W, requires_grad=True)      # 平预测
    d = torch.rand(B, H, W)                            # 非平目标
    mask = torch.ones(B, H, W)
    assert multiscale_grad(mu, d, mask).item() > 0


def test_multiscale_grad_mask_aware_pooling():
    # 掩码感知降采样正确性：无效像素不得参与池化——改无效像素的值，损失不变
    mu, _, d, mask, _, _ = _batch()
    mask[:, ::2, ::2] = 0.0
    d2 = d.clone(); d2[mask == 0] = 1e6
    assert torch.allclose(multiscale_grad(mu, d, mask),
                          multiscale_grad(mu, d2, mask), atol=1e-6)


def test_multiscale_grad_nan_masked_safe():
    mu, _, d, mask, _, _ = _batch()
    mask[:, :16] = 0.0
    d[:, :16] = float("nan")
    assert torch.isfinite(multiscale_grad(mu, d, mask))


# ---- L_smooth：仅空洞区 ----

def test_edge_smooth_only_holes():
    mu, _, _, mask, _, rgb = _batch()
    mask[:, :32] = 0.0                         # 上半空洞
    base = edge_smooth(mu, rgb, mask)
    with torch.no_grad():
        mu2 = mu.clone(); mu2[:, 32:] = 5.0    # 只改有效区 → 损失不变
    assert torch.allclose(base, edge_smooth(mu2, rgb, mask))


def test_edge_smooth_flat_is_zero():
    mu = torch.ones(B, H, W, requires_grad=True)
    mask = torch.zeros(B, H, W)                # 全空洞
    rgb = torch.rand(B, 3, H, W)
    assert edge_smooth(mu, rgb, mask).item() < 1e-7


def test_edge_smooth_rgb_downsampled_internally():
    mu = torch.rand(B, H, W, requires_grad=True)
    mask = torch.zeros(B, H, W)
    rgb_full = torch.rand(B, 3, 224, 224)      # 允许传满分辨率 RGB
    assert torch.isfinite(edge_smooth(mu, rgb_full, mask))


# ---- 组合 DepthLoss ----

def test_depth_loss_composition():
    mu, log_sigma, d, mask, teacher, rgb = _batch()
    mask[:, :20] = 0.0
    crit = DepthLoss()
    out = crit(mu, log_sigma, d, mask, teacher, rgb)
    assert set(out) == {"total", "metric", "teacher", "grad", "smooth"}
    expected = (1.0 * out["metric"] + 1.0 * out["teacher"]
                + 0.5 * out["grad"] + 0.1 * out["smooth"])
    assert torch.allclose(out["total"], expected)
    out["total"].backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(log_sigma.grad).all()


def test_depth_loss_empty_mask_still_finite():
    mu, log_sigma, d, mask, teacher, rgb = _batch(mask_fill=0.0)
    out = DepthLoss()(mu, log_sigma, d, mask, teacher, rgb)
    assert torch.isfinite(out["total"])
    out["total"].backward()


# ---- L_dyn：动力学预测 ----

def test_dyn_loss_basic_and_target_detached():
    z_hat = torch.randn(2, 256, 384, requires_grad=True)
    z_target = torch.randn(2, 256, 384, requires_grad=True)
    loss = latent_prediction_loss(z_hat, z_target)
    loss.backward()
    assert z_hat.grad.abs().sum() > 0
    assert z_target.grad is None or z_target.grad.abs().sum() == 0


def test_dyn_loss_zero_when_perfect():
    z = torch.randn(2, 256, 384)
    assert latent_prediction_loss(z, z.clone()).item() == 0.0


def test_dyn_loss_dynamic_weighting():
    torch.manual_seed(0)
    z_hat = torch.zeros(1, 4, 8)
    z_target = torch.ones(1, 4, 8)
    z_target[0, 0] = 2.0                       # patch 0 误差更大
    z_prev = torch.zeros(1, 4, 8)
    z_prev[0, 0] = 10.0                        # patch 0 是动态区
    w_dyn = latent_prediction_loss(z_hat, z_target, z_prev, dyn_weight=True)
    w_flat = latent_prediction_loss(z_hat, z_target, z_prev, dyn_weight=False)
    # 动态加权把权重集中到动态 patch（误差更大处）⇒ 加权损失高于均匀加权
    assert w_dyn.item() > w_flat.item()
