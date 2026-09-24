#!/usr/bin/env python
"""录制/训练前置冒烟：单进程单 env（IsaacLab 同进程建第二个 env 生命周期
脆弱，2026-09-17 实测会卡在 gym.make——故拆两个模式分两次跑）。

用法（env_isaaclab）：
    python src/simulation/scripts/smoke_record_setup.py --mode camera
    python src/simulation/scripts/smoke_record_setup.py --mode rl

camera 模式验证：三相机建成、随机化后 K round-trip（请求 fx vs 实测
fx 应一致）、基座原点（应 ≈[0,0.15,0.75]）、T_base_cam 量级合理、
absolute_joint16 输出。
rl 模式验证：make_rl_cfg 无相机场景能建 env、policy 观测维度正常。
"""

import argparse
import sys
from pathlib import Path

_SIM_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _SIM_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))

# 注意顺序：SimContext 必须在任何 simulation.tasks / isaaclab 导入之前进入
from simulation.sim_context import SimContext  # noqa: E402


def mode_camera():
    import numpy as np
    import torch

    with SimContext(headless=True, enable_cameras=True) as ctx:
        from simulation import recording as rec
        from simulation.tasks.hang_cup.env_cfg import (
            camera_names, load_sim_config)

        cfg = load_sim_config()
        env = ctx.make_env(cfg["task_id"], num_envs=1)
        obs, _ = env.reset()
        names = camera_names()
        print("相机成员:", names, "->",
              list(env.unwrapped.scene.sensors.keys()))

        poses = rec.sample_camera_poses(cfg["cameras"],
                                        np.random.default_rng(0))
        rec.randomize_cameras(env, poses)
        home = torch.zeros(1, 16, device=env.unwrapped.device)
        for _ in range(3):                       # 等渲染管线刷新
            env.step(home)

        Twb = rec.compute_base_frame(env)
        print("基座原点（应≈[0,0.15,0.75]）:", np.round(Twb[:3, 3], 4))
        for n in names:
            cam = env.unwrapped.scene.sensors[f"camera_{n}"]
            K, T = rec.camera_extrinsics_in_base(cam, Twb)
            K_req = rec.pinhole_K(cam.cfg.width, cam.cfg.height,
                                  poses[n]["focal_mm"])
            print(f"{n}: fx 请求 {K_req[0, 0]:.2f} vs 实测 {K[0, 0]:.2f}; "
                  f"T_base_cam t={np.round(T[:3, 3], 3)}")
        print("abs16:", np.round(rec.absolute_joint16(env), 3))
    print("SMOKE camera OK")


def mode_rl():
    with SimContext(headless=True, enable_cameras=False) as ctx:
        from simulation.tasks.hang_cup.env_cfg import (
            load_sim_config, make_rl_cfg)

        cfg = make_rl_cfg(4)
        env = ctx.make_env(load_sim_config()["task_id"], cfg=cfg)
        obs, _ = env.reset()
        print("RL cfg ok: num_envs=4, sensors =",
              list(env.unwrapped.scene.sensors.keys()),
              ", policy dim =",
              env.unwrapped.observation_manager.group_obs_dim["policy"])
    print("SMOKE rl OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["camera", "rl"], required=True)
    args = ap.parse_args()
    {"camera": mode_camera, "rl": mode_rl}[args.mode]()


if __name__ == "__main__":
    main()
