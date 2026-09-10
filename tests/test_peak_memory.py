"""最悲观显存测试：全部模块同时在卡 + 联合微调图 + 多步展开 + GRL + 优化器状态。

对应 guideline §9 的 Stage 2 预算（≤13GB，16GB 卡）：E 可训练、防漂移锚
（冻结 E 副本）前向、T 多步展开不截断梯度、D 解码每一帧 ᑮ、判别器挂在
z_t 与所有 ᑮ 上、AdamW 状态实际分配（step 之后才占显存）。

这是"最坏情况"上界：Stage 0（无 T/判别器、无展开）与 Stage 1（E 冻结、
读缓存 latent）的显存都严格低于本场景。峰值超过预算 = 必须按 §9 降载顺序
调整，测试失败即预警。
"""

import pytest
import torch

from conftest import ENCODER_PATH

from sawvla.losses import DepthLoss, grad_reverse, latent_prediction_loss  # noqa: E402
from sawvla.models import (ActionAdapter, DINOv2Encoder,  # noqa: E402
                           DepthDecoder, DomainDiscriminator, TransitionModel)

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU"),
              pytest.mark.skipif(not ENCODER_PATH.is_dir(),
                                 reason="编码器权重未下载")]

BATCH = 8            # §9 Stage 2：batch 紧张时减半累积，这里取偏保守的 8
N_UNROLL = 4         # 多步展开 k∈{1,2,4} 的最大档
PEAK_BUDGET_GIB = 13.0


def test_peak_memory_all_modules_worst_case():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dev = "cuda"

    E = DINOv2Encoder(ENCODER_PATH).to(dev)                    # 可训练
    E_anchor = DINOv2Encoder(ENCODER_PATH).to(dev)             # 防漂移锚
    E_anchor.freeze()
    adapter = ActionAdapter(action_dim=14, proprio_dim=16).to(dev)
    T = TransitionModel().to(dev)                              # 完整 8 层
    D = DepthDecoder().to(dev)
    disc = DomainDiscriminator().to(dev)
    crit = DepthLoss()

    params = [p for m in (E, adapter, T, D, disc)
              for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)

    n_frames = N_UNROLL + 1
    rgb = torch.randn(BATCH, n_frames, 3, 224, 224, device=dev)
    actions = torch.randn(BATCH, N_UNROLL, 14, device=dev)     # 逐步动作
    proprio = torch.randn(BATCH, N_UNROLL, 16, device=dev)
    d_tgt = torch.rand(BATCH, 64, 64, device=dev)
    mask = (torch.rand(BATCH, 64, 64, device=dev) > 0.2).float()
    teacher = torch.rand(BATCH, 64, 64, device=dev)
    domain = (torch.rand(BATCH * (N_UNROLL + 1), device=dev) > 0.5).float()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # 所有帧过 E（联合微调，E 携带梯度图——比 Stage 1 缓存方案更吃显存）
        z = [E(rgb[:, i]) for i in range(n_frames)]
        with torch.no_grad():
            z_anchor = E_anchor(rgb[:, 0])
        # 防漂移蒸馏正则（小权重，§2.3-E）
        loss = 0.1 * torch.nn.functional.mse_loss(z[0], z_anchor)

        # teacher-forced block-causal 一次前向算 k 步预测（每步解码，比
        # stop-grad 更悲观）
        z_seq = torch.stack(z[:N_UNROLL], dim=1)               # (B, k, 256, 384)
        act_tok, prop_tok = adapter(actions, proprio)
        z_hat = T(z_seq, act_tok, prop_tok)                    # (B, k, 256, 384)

        adv_logits = [disc(grad_reverse(z[0], 0.05))]
        for k in range(N_UNROLL):
            loss = loss + latent_prediction_loss(z_hat[:, k], z[k + 1],
                                                 z_prev=z[k])
            mu, log_sigma = D(z_hat[:, k])
            loss = loss + crit(mu, log_sigma, d_tgt, mask, teacher,
                               rgb[:, k + 1])["total"]
            adv_logits.append(disc(grad_reverse(z_hat[:, k], 0.05)))

        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat(adv_logits), domain)

    opt.zero_grad()
    loss.backward()
    opt.step()                                   # 强制 AdamW m/v 状态实际分配
    torch.cuda.synchronize()

    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    n_params = sum(p.numel() for p in params) / 1e6
    print(f"\n[peak-memory] batch={BATCH} unroll={N_UNROLL} "
          f"trainable={n_params:.1f}M peak={peak_gib:.2f} GiB "
          f"(预算 {PEAK_BUDGET_GIB} GiB)")
    assert peak_gib < PEAK_BUDGET_GIB, (
        f"峰值显存 {peak_gib:.2f} GiB 超 Stage 2 预算 {PEAK_BUDGET_GIB} GiB；"
        f"按 §9 降载：先减 batch → 再降深度输出分辨率 → 再降输入分辨率")
