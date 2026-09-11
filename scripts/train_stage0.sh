#!/usr/bin/env bash
# Stage 0 正式训练（E + D 深度塑形，mask-only 基线）。
# 超参依据 guideline §5-Stage 0：AdamW lr 1e-4，batch 32（累积等效 64），
# ~100K 步，bf16；防漂移锚与在线 disparity 监督见 train_stage0.py 模块头。
# 预计 4–8h（4070 Ti S，guideline §9.1 估算量级）。
#
# 用法：conda activate style_agnostic_wam && bash scripts/train_stage0.sh
# 监控：cd 到本仓库根目录后 tensorboard --logdir outputs/stage0/tb
# （tensorboard 只装在 style_agnostic_wam 环境里，需先 conda activate;  --logdir 是相对路径）
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/train_stage0.py \
    --steps 100000 \
    --batch 32 \
    --accum 2 \
    --lr 1e-4 \
    --workers 4 \
    --val-every 1000 \
    --outdir outputs/stage0
