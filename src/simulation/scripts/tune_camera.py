#!/usr/bin/env python
"""相机机位调参工具：用命令行给的 pos/look-at 覆盖 YAML，渲一帧存图。

近似 real 0520 批斜视机位用（2026-09-13 裁决 #4：先近似，保留调整工具）。

用法：
    python src/simulation/scripts/tune_camera.py \
        [--camera front] --pos 0,-0.95,1.45 --look-at 0,0.05,0.75 [--out /tmp/cam.png]
反复调整参数直到渲染视角与 real 参考帧
（outputs/check_franka/ 或 check_camera_views/ 里的 real 图）对齐，
然后把最终参数写回 src/simulation/config/simulation.yaml 对应相机条目。
"""

import argparse
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC_ROOT))

from simulation.sim_context import SimContext  # noqa: E402


def parse_xyz(s: str) -> list[float]:
    v = [float(x) for x in s.split(",")]
    assert len(v) == 3
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="front",
                    help="YAML cameras 下的键名（front/left/right/...）")
    ap.add_argument("--pos", type=parse_xyz, default=None)
    ap.add_argument("--look-at", type=parse_xyz, default=None)
    ap.add_argument("--focal", type=float, default=None, help="焦距 mm")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/sim_play/tune_camera.png"))
    args = ap.parse_args()

    import cv2
    import torch

    with SimContext(headless=True) as ctx:
        from simulation.tasks.hang_cup.env_cfg import load_sim_config
        cfg = load_sim_config()
        cam_key = args.camera
        if args.pos is not None:
            cfg["cameras"][cam_key]["pos"] = args.pos
        if args.look_at is not None:
            cfg["cameras"][cam_key]["look_at"] = args.look_at
        if args.focal is not None:
            cfg["cameras"][cam_key]["focal_length_mm"] = args.focal

        # 用覆盖后的配置现场构建 env cfg
        import simulation.tasks.hang_cup.env_cfg as ec
        ec.load_sim_config = lambda: cfg          # 注入覆盖（仅本进程）
        env = ctx.make_env(cfg["task_id"], num_envs=1)
        env.reset()
        home = torch.zeros(1, 16, device=env.unwrapped.device)
        for _ in range(10):                       # 等渲染管线稳定
            env.step(home)
        cam = env.unwrapped.scene.sensors[f"camera_{cam_key}"]
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        from simulation.viz import read_depth_mm
        dep = read_depth_mm(cam)                              # (H,W) mm

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    # 深度伪彩色与数据侧（check_franka_dataset.depth_to_rgb）同一逻辑
    from simulation.viz import depth_to_rgb
    d = (depth_to_rgb(dep) * 255).astype("uint8")
    cv2.imwrite(str(args.out.with_name(args.out.stem + "_depth.png")),
                cv2.cvtColor(d, cv2.COLOR_RGB2BGR))
    print(f"camera={cam_key} pos={cfg['cameras'][cam_key]['pos']} "
          f"look_at={cfg['cameras'][cam_key]['look_at']} "
          f"focal={cfg['cameras'][cam_key]['focal_length_mm']}")
    print(f"-> {args.out}（及 _depth.png）")


if __name__ == "__main__":
    main()
