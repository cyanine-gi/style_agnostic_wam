#!/usr/bin/env python
"""hang_cup 一键冒烟：启动仿真 → 建 env → 默认姿态（home）若干部 →
随机动作若干 episode → 保存全局相机 RGB/深度网格到 outputs/sim_play/。

用途：验证"环境能建、物理能跑、视觉在环、16 维动作契约生效"。
不接策略（RL 训练脚本另起，见 rl/facade.py 包装层）。

用法（需要含 isaaclab 的环境，如 env_isaaclab 或新建的合并环境）：
    python src/simulation/scripts/play_hang_cup.py [--steps 150] [--episodes 2]
"""

import argparse
import sys
from pathlib import Path

_SIM_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _SIM_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))

# 注意顺序：SimContext 必须在任何 simulation.tasks / isaaclab 导入之前进入
from simulation.sim_context import SimContext  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=150,
                    help="home 姿态保持步数（其后随机动作）")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--random-steps", type=int, default=100)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/sim_play"))
    args = ap.parse_args()

    import numpy as np
    import torch
    import cv2

    args.outdir.mkdir(parents=True, exist_ok=True)

    with SimContext(headless=args.headless) as ctx:
        from simulation.tasks.hang_cup.env_cfg import load_sim_config
        task_id = load_sim_config()["task_id"]
        env = ctx.make_env(task_id, num_envs=1)
        print(f"env: {task_id}, action_dim={env.unwrapped.action_manager.total_action_dim}, "
              f"obs policy dim={env.unwrapped.observation_manager.group_obs_dim['policy']}")

        obs, _ = env.reset()
        frames = []

        def snapshot(step, tag):
            from simulation.viz import read_depth_mm
            cam = env.unwrapped.scene.sensors["camera_front"]
            rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
            dep = read_depth_mm(cam)                          # (H,W) mm
            frames.append((step, tag, rgb, dep))

        # 阶段 1：保持姿态（delta 动作契约：全零 = 保持当前关节位，
        # 2026-09-18 起动作项为增量式，reset 后当前位即 home）
        home = torch.zeros(1, 16, device=env.unwrapped.device)
        for t in range(args.steps):
            obs, rew, term, trunc, _ = env.step(home)
            if t % 30 == 0:
                snapshot(t, "home")

        # 阶段 2：随机动作 episode（raw delta，验证动作链路 + 物理响应）
        rng = torch.Generator(device=env.unwrapped.device).manual_seed(0)
        for ep in range(args.episodes):
            obs, _ = env.reset()
            for t in range(args.random_steps):
                a = 0.3 * torch.randn(
                    home.shape, generator=rng, device=env.unwrapped.device)
                obs, rew, term, trunc, _ = env.step(a)
                if t % 25 == 0:
                    snapshot(t, f"rand_ep{ep}")
            print(f"episode {ep}: 结束 term={term.any().item()} "
                  f"trunc={trunc.any().item()}")

    # 存网格图（深度伪彩色与数据侧 check_franka_dataset.depth_to_rgb 同逻辑）
    from simulation.viz import depth_to_rgb
    n = len(frames)
    cols = 3
    rows = (n + cols - 1) // cols
    th, tw = 360, 640
    grid = np.zeros((rows * th * 2, cols * tw, 3), np.uint8)
    for i, (step, tag, rgb, dep) in enumerate(frames):
        r, c = divmod(i, cols)
        im = cv2.resize(rgb, (tw, th))
        dm = cv2.resize((depth_to_rgb(dep) * 255).astype(np.uint8), (tw, th))
        cv2.putText(im, f"{tag} step{step}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        grid[(r * 2) * th:(r * 2 + 1) * th, c * tw:(c + 1) * tw] = im
        grid[(r * 2 + 1) * th:(r * 2 + 2) * th, c * tw:(c + 1) * tw] = dm
    out = args.outdir / "play_grid.png"
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
