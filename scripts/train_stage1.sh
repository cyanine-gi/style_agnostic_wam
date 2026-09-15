#!/usr/bin/env bash
# Stage 1 正式训练（动作条件转移模型 T，register 递推，自由展开 K=4）。
# 超参依据 guideline §6-Stage 1（2026-09-15 裁决定稿）：lr 1e-4，~150K 步，
# batch 8（旧估值区间 8–16；实测 batch 8 显存仅 ~1.8GB，富余可 --batch 16，
# 但本负载下更大 batch 不省 wall-clock），bf16。
# 前置：outputs/latent_cache 已构建（scripts/build_latent_cache.py，
# 619 eps / 8.4 万帧，real 15fps / sim 抽帧到 15fps）。
# D 与信号头从 outputs/stage0/last.pt 初始化，T/adapter 从零。
# 预计 ~7h（4070 Ti S 实测 ~0.17s/step @ batch 8）。
#
# 用法：conda activate style_agnostic_wam && bash scripts/train_stage1.sh
# 监控：cd 到本仓库根目录后 tensorboard --logdir outputs/stage1/tb
# （tensorboard 只装在 style_agnostic_wam 环境里，需先 conda activate;  --logdir 是相对路径）
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/train_stage1.py \
    --steps 150000 \
    --batch 8 \
    --workers 4 \
    --val-every 1000 \
    --outdir outputs/stage1
