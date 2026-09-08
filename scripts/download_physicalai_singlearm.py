#!/usr/bin/env python
"""从 ModelScope 下载 PhysicalAI-Robotics-Manipulation-SingleArm（Franka 仿真，整仓 ~15.3GB）。

用法：
  python scripts/download_physicalai_singlearm.py     # 整仓（6 个子数据集全下）
"""

from pathlib import Path

from modelscope import snapshot_download

REPO = "nv-community/PhysicalAI-Robotics-Manipulation-SingleArm"
OUT = Path("data/physicalai_singlearm")

if __name__ == "__main__":
    print(f"下载 {REPO} -> {OUT}")
    snapshot_download(REPO, repo_type="dataset", local_dir=str(OUT), max_workers=8)
    print(f"完成 -> {OUT}")
