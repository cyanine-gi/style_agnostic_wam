#!/usr/bin/env bash
# 用 huggingface_hub 官方 pip 包，经 hf-mirror.com 镜像下载
# nvidia/PhysicalAI-Robotics-mindmap-GR1-Stick-in-Bin
# （Fourier GR1 人形双臂，Isaac Lab 遥操作 + Mimic 扩增，mindmap 格式含逐相机深度 PNG + 内参）
# 整仓 ~102.5GB：10 个 demo tar（每个 ~9-11GB）+ 200 条演示的 HDF5，到 data/ 下。
#
# 注意：ModelScope 上的 nv-community 同名仓库是空壳（只有 README，无数据），
#       只能从 HF 经 hf-mirror 下载。
#
# 说明：
#   - 仓库仅 13 个文件，无需翻页；但 hf-mirror 对匿名请求有限流（429），
#     列表与下载都带退避重试。
#   - hf_hub_download 自带临时文件续传，中断后重跑可续。
#
# 用法：
#   bash scripts/download_mindmap_gr1_stick_in_bin.sh
# 断点续传：中断后重新运行同一命令即可。公开仓库，无需登录。

set -euo pipefail

REPO="nvidia/PhysicalAI-Robotics-mindmap-GR1-Stick-in-Bin"
OUT="data/mindmap_gr1_stick_in_bin"

# 确保官方工具就绪
python -c "import huggingface_hub" 2>/dev/null || pip install -U huggingface_hub

mkdir -p "$OUT"

echo "下载 ${REPO} -> ${OUT}（endpoint: https://hf-mirror.com）"

HF_ENDPOINT="https://hf-mirror.com" HF_HUB_DISABLE_XET=1 REPO="$REPO" OUT="$OUT" python - <<'EOF'
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = os.environ["REPO"]
OUT = Path(os.environ["OUT"])

# 列出仓库文件（仅 13 个，单页即可；限流时退避重试）
api = HfApi(endpoint=os.environ["HF_ENDPOINT"])
for attempt in range(20):
    try:
        files = api.list_repo_files(REPO, repo_type="dataset")
        break
    except Exception as e:
        if "429" not in str(e):
            raise
        wait = min(30 * (attempt + 1), 180)
        print(f"  限流(429)，{wait}s 后重试（第 {attempt + 1} 次）...", flush=True)
        time.sleep(wait)
else:
    raise RuntimeError("连续 20 次 429，镜像限流未解除，稍后再运行本脚本")

print(f"共 {len(files)} 个文件，开始下载（已存在且完整的自动跳过）...")


def download_one(path):
    if (OUT / path).exists():  # 已完成的文件直接跳过
        return True
    for attempt in range(10):
        try:
            hf_hub_download(REPO, path, repo_type="dataset", local_dir=str(OUT))
            return True
        except Exception as e:
            if "429" in str(e):
                wait = min(30 * (attempt + 1), 180)
                time.sleep(wait)
                continue
            print(f"  失败（跳过）{path}: {e}", flush=True)
            return False
    print(f"  持续限流（跳过）{path}", flush=True)
    return False


ok = 0
for i, path in enumerate(files, 1):
    print(f"[{i}/{len(files)}] {path}", flush=True)
    ok += download_one(path)

print(f"完成 -> {OUT}（成功 {ok}/{len(files)}）")
if ok < len(files):
    print("有文件未完成，重新运行本脚本即可续传")
EOF
