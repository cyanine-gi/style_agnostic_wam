#!/usr/bin/env bash
# 用 modelscope 官方 pip 包，从 ModelScope 下载
# nv-community/PhysicalAI-Robotics-mindmap-GR1-Drill-in-Box
# （Fourier GR1 人形双臂，Isaac Lab 遥操作 + Mimic 扩增，mindmap 格式含逐相机深度 PNG + 内参）
# 整仓 ~53.7GB：10 个 demo tar（每个 ~5-6GB）+ 200 条演示的 HDF5，到 data/ 下。
#
# 说明：
#   - 仓库仅 13 个文件，无翻页/限流问题（不同于 hf-mirror 下载 SingleArm 的场景），
#     直接按文件逐个下载，失败退避重试。
#   - modelscope SDK 的 dataset_file_download 自带临时文件续传，中断后重跑可续。
#
# 用法：
#   bash scripts/download_mindmap_gr1_drill_in_box.sh
# 断点续传：中断后重新运行同一命令即可。公开仓库，无需登录。

set -euo pipefail

REPO="nv-community/PhysicalAI-Robotics-mindmap-GR1-Drill-in-Box"
OUT="data/mindmap_gr1_drill_in_box"

# 确保官方工具就绪
python -c "import modelscope" 2>/dev/null || pip install -U modelscope

mkdir -p "$OUT"

echo "下载 ${REPO} -> ${OUT}（ModelScope）"

REPO="$REPO" OUT="$OUT" python - <<'EOF'
import os
import time
from pathlib import Path

import requests
from modelscope.hub.file_download import dataset_file_download

REPO = os.environ["REPO"]
OUT = Path(os.environ["OUT"])

# 列出仓库文件（仅 13 个，无需缓存/翻页）
r = requests.get(
    f"https://modelscope.cn/api/v1/datasets/{REPO}/repo/tree",
    params={"Revision": "master", "Recursive": "true"},
    timeout=60,
)
r.raise_for_status()
# 注意：该 API 的文件类型字段是 git 风格的 "blob"/"tree"，不是 "file"
files = [f["Path"] for f in r.json()["Data"]["Files"] if f.get("Type") == "blob"]
print(f"共 {len(files)} 个文件，开始下载（已存在且完整的自动跳过）...")


def download_one(path):
    for attempt in range(10):
        try:
            dataset_file_download(REPO, path, local_dir=str(OUT))
            return True
        except Exception as e:
            wait = min(30 * (attempt + 1), 180)
            print(f"  失败 {path}: {e}，{wait}s 后重试（第 {attempt + 1} 次）...", flush=True)
            time.sleep(wait)
    print(f"  持续失败（跳过）{path}", flush=True)
    return False


ok = 0
for i, path in enumerate(files, 1):
    print(f"[{i}/{len(files)}] {path}", flush=True)
    ok += download_one(path)

print(f"完成 -> {OUT}（成功 {ok}/{len(files)}）")
if ok < len(files):
    print("有文件未完成，重新运行本脚本即可续传")
EOF
