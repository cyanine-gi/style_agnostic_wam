#!/usr/bin/env bash
# 用 huggingface_hub 官方 pip 包，经 hf-mirror.com 镜像下载
# nvidia/PhysicalAI-Robotics-Manipulation-SingleArm（Franka 仿真，整仓 ~15.3GB、13.6 万文件）到 data/ 下。
#
# 两个坑及对策：
#   1) 仓库文件数 >1000，HF 列表 API 需翻页；hf-mirror 不改写分页 Link 头里的
#      官网绝对地址，裸用会直连 huggingface.co（国内不可达）。
#      -> 这里自己翻页，并把下一页地址重写回镜像。
#   2) hf-mirror 对匿名 API 有限流（HTTP 429），137 页列表容易触发。
#      -> 列表结果缓存到 .file_list.json，每页落盘；429 时退避等待重试，
#         中断后重跑从上次页码继续；列表完成后重跑直接用缓存，不再请求列表 API。
#
# 用法：
#   bash scripts/download_physicalai_singlearm.sh
# 断点续传：中断后重新运行同一命令即可（列表和文件下载都可续）。公开仓库，无需登录。

set -euo pipefail

REPO="nvidia/PhysicalAI-Robotics-Manipulation-SingleArm"
OUT="data/physicalai_singlearm"

# 确保官方工具就绪
python -c "import huggingface_hub" 2>/dev/null || pip install -U huggingface_hub

mkdir -p "$OUT"

echo "下载 ${REPO} -> ${OUT}（endpoint: https://hf-mirror.com）"

HF_ENDPOINT="https://hf-mirror.com" HF_HUB_DISABLE_XET=1 REPO="$REPO" OUT="$OUT" python - <<'EOF'
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import get_session

ENDPOINT = os.environ["HF_ENDPOINT"].rstrip("/")
REPO = os.environ["REPO"]
OUT = Path(os.environ["OUT"])
STATE = OUT / ".file_list.json"   # 列表缓存（含翻页游标，可断点续传）
WORKERS = 4                        # 镜像限流，别开太高


def get_with_backoff(url, params):
    """GET 带耐心的 429 退避：镜像限流窗口约 1 分钟，官方默认重试太短。"""
    for attempt in range(20):
        r = get_session().get(url, params=params, timeout=60)
        if r.status_code == 429:
            wait = min(30 * (attempt + 1), 180)
            print(f"  限流(429)，{wait}s 后重试（第 {attempt + 1} 次）...", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("连续 20 次 429，镜像限流未解除，稍后再运行本脚本")


def list_files():
    """翻页列出全部文件，每页落盘，中断后从上次游标继续。"""
    sha = HfApi(endpoint=ENDPOINT).repo_info(REPO, repo_type="dataset").sha

    cursor, files = None, []
    if STATE.exists():
        state = json.loads(STATE.read_text())
        if state.get("sha") == sha:
            cursor, files = state["cursor"], state["files"]
            if cursor is None:
                print(f"使用缓存的文件列表（{len(files)} 个文件）")
                return files
            print(f"续传列表：已有 {len(files)} 个文件，从第 {len(files) // 1000 + 1} 页继续")

    url = f"{ENDPOINT}/api/datasets/{REPO}/tree/{sha}"
    while True:
        params = {"recursive": "true", "expand": "false", "limit": "1000"}
        if cursor:
            params["cursor"] = cursor
        r = get_with_backoff(url, params)
        files += [f["path"] for f in r.json() if f["type"] == "file"]

        next_url = r.links.get("next", {}).get("url")
        # hf-mirror 不改写分页头，下一页是官网绝对地址，重写回镜像取游标
        cursor = parse_qs(urlparse(next_url).query)["cursor"][0] if next_url else None
        STATE.write_text(json.dumps({"sha": sha, "cursor": cursor, "files": files}))
        print(f"  已列出 {len(files)} 个文件...", flush=True)
        if cursor is None:
            return files


def download_one(path):
    dest = OUT / path
    if dest.exists():  # 已完成的文件直接跳过
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


files = list_files()
print(f"共 {len(files)} 个文件，开始下载（{WORKERS} 线程，已存在的自动跳过）...")

done = ok = 0
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for ok_one in pool.map(download_one, files):
        done += 1
        ok += ok_one
        if done % 500 == 0:
            print(f"  进度 {done}/{len(files)}", flush=True)

print(f"完成 -> {OUT}（成功 {ok}/{len(files)}）")
if ok < len(files):
    print("有文件未完成，重新运行本脚本即可续传")
EOF
