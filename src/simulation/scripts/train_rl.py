#!/usr/bin/env python
"""hang_cup RL 训练（rsl_rl PPO，纯状态策略）。

2026-09-17 裁决（两阶段之阶段一）：纯状态观测（proprio + 杯/钉位姿 +
关节速度），不渲染相机——最快收敛、解锁录制；像素策略（固定俯视相机 +
CNN encoder）是阶段二后续实验，不在本脚本。

用法（需要 env_isaaclab 环境）：
    python src/simulation/scripts/train_rl.py \
        [--num-envs 64] [--max-iterations 3000] [--log-dir outputs/rl/hang_cup]
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
    ap.add_argument("--num-envs", type=int, default=None,
                    help="并行环境数，默认取 simulation.yaml env.rl_num_envs")
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--log-dir", type=Path, default=Path("outputs/rl/hang_cup"))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    with SimContext(headless=True, enable_cameras=False) as ctx:
        import simulation.tasks  # noqa: F401  （触发 gym.register）
        from simulation.rl import facade
        from simulation.tasks.hang_cup.env_cfg import (
            load_sim_config, make_rl_cfg)
        from simulation.tasks.hang_cup.agents.rsl_rl_ppo_cfg import (
            HangCupPPORunnerCfg)

        sim_cfg = load_sim_config()
        cfg = make_rl_cfg(args.num_envs)
        env = ctx.make_env(sim_cfg["task_id"], cfg=cfg)
        print(f"env: {sim_cfg['task_id']}, num_envs={cfg.scene.num_envs}, "
              f"obs policy dim="
              f"{env.unwrapped.observation_manager.group_obs_dim['policy']}")

        wrapped = facade.wrap_env_for_training(env)
        agent_cfg = HangCupPPORunnerCfg()
        if args.max_iterations is not None:
            agent_cfg.max_iterations = args.max_iterations
        if args.seed is not None:
            agent_cfg.seed = args.seed
        args.log_dir.mkdir(parents=True, exist_ok=True)
        runner = facade.make_runner(wrapped, agent_cfg,
                                    log_dir=str(args.log_dir))
        runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                     init_at_random_ep_len=True)
        print(f"-> {args.log_dir}（策略 ckpt 在各 run 目录的 model.pt）")


if __name__ == "__main__":
    main()
