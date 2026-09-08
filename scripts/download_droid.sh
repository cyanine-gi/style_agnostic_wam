#!/usr/bin/env bash
# 下载 DROID LeRobot 移植版（cadene/droid_1.0.1）的同任务子集——只取 drawer/cabinet/stack 三族
#（与 PhysicalAI-SingleArm 仿真侧任务一一对应，见 docs/data_preparation.md §2）。
#
# 与 download_physicalai_singlearm.sh 相同的网络处境与对策：
#   - 本机直连 huggingface.co 不通 -> 文件走 hf-mirror 镜像（hf_utils.py 默认已设）；
#   - hf-mirror 对 API 限流（429）-> 本脚本完全不用文件列举 API：
#     先只下 meta/ 小文件（info.json + episodes.jsonl，~20MB），按语言指令关键词
#     在本地过滤出三族 episode，再按 info.json 的路径模板逐 episode 精准下载
#     parquet + exterior_1_left 一路视频。断点续传：中断后重跑即可，已下文件自动跳过。
#
# 用法：
#   bash scripts/download_droid.sh            # 三族全量（命中约 8k episodes，~36GB）
#   TARGET=5000 bash scripts/download_droid.sh # 可选：种子可复现子采样到 5000 条
#   DRY_RUN=1 bash scripts/download_droid.sh   # 只出 matching_report.json，不下载视频/parquet
#
# 下载前请人工抽检输出目录 matching_report.json 里的示例指令（data_preparation.md §2）。

set -euo pipefail
cd "$(dirname "$0")/.."   # 保证从仓库根运行（src/ 与 configs/ 的相对路径生效）

REPO="cadene/droid_1.0.1"
OUT="data/droid_subset"
TARGET="${TARGET:-}"
DRY_RUN="${DRY_RUN:-}"

# 确保依赖就绪（官方 pip 包；yaml/tqdm 为项目过滤脚本所需）
python -c "import huggingface_hub, yaml, tqdm" 2>/dev/null || pip install -U huggingface_hub pyyaml tqdm

# 只保留 drawer/cabinet/stack 三族的关键词表（从 configs/droid_task_keywords.yaml 去掉过宽的 pick_place；
# 落盘到输出目录，便于追溯本次过滤用的确切规则）
mkdir -p "$OUT"
KEYWORDS="$OUT/filter_keywords_drawer_cabinet_stack.yaml"
cat > "$KEYWORDS" <<'YAML'
# 由 scripts/download_droid.sh 生成：仅 drawer/cabinet/stack 三族
families:
  drawer:
    - "drawer"
  cabinet:
    - "cabinet"
    - "cupboard"
  stack:
    - "stack"
YAML

echo "下载 ${REPO} 的 drawer/cabinet/stack 子集 -> ${OUT}"

ARGS=(--filter-tasks "$KEYWORDS" --out "$OUT" --workers 4)
[ -n "$TARGET" ] && ARGS+=(--target "$TARGET")
[ -n "$DRY_RUN" ] && ARGS+=(--dry-run)

python src/sawvla/data/download_droid.py "${ARGS[@]}"
