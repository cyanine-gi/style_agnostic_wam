#!/usr/bin/env bash
# Stage 2 正式训练（E+T 联合在线微调 + 消灭 reg_agnostic 域信息）。
# 依据 guideline §6-Stage 2 + 2026-09-16/17 三次裁决：联合在线（E 动了缓存
# 即过期，clip 从原始 HDF5 在线编码）、锚缩到 patch+reg_domain、batch 精确
# 50/50 域均衡。
# [2026-09-17 裁决·三] 清洗目标双开关（configs/model.yaml stage2.adv）：
# 默认 HSIC 开（λ=1.0，ramp 10k，跨步缓冲 256；无对弈）+ 熵混淆关。
# 判别器只在熵混淆开启时建造。val 探针独立（现场从零训，基线=经验先验
# val/probe_prior），逐域分层取样。失败复盘见 src/sawvla/train_failed_log.md。
# 初始化：E <- outputs/stage0/last.pt，T/adapter/D/signals <- outputs/stage1/last.pt。
# 实测 ~0.4s/step @ batch 8 / workers 8，20K 步 ≈ 2h；
# 可用 --resume outputs/stage2/ckpt_stepXXXXXX.pt 断点续训。
#
# 用法：conda activate style_agnostic_wam && bash scripts/train_stage2.sh
# 监控：cd 到本仓库根目录后 tensorboard --logdir outputs/stage2/tb
# （tensorboard 只装在 style_agnostic_wam 环境里，需先 conda activate;  --logdir 是相对路径）
# 关键盯：val/probe_e/reg_agnostic 应降到 ≈0.5（验收门）、train/disc/acc_*、
# val/depth/true 不应退化超过 5%（退化即回调 λ，见 guideline §6）。
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/train_stage2.py \
    --steps 20000 \
    --batch 8 \
    --workers 8 \
    --lr 5e-5 \
    --val-every 500 \
    --outdir outputs/stage2
