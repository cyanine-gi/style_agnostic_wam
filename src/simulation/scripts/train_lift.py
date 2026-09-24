#!/usr/bin/env python
"""lift_cup RL 训练（rsl_rl PPO，Isaac-Lift 原生配方，纯状态策略）。

2026-09-21 用户裁决新建：同场景最小化任务——右手抓杯举离桌面 15cm，
不挂钉、无状态机。任务 Saw-LiftCup-FrankaDual-v0（MDP:
tasks/hang_cup/mdp_lift.py，装配: lift_env_cfg.py）。

用法（需要 env_isaaclab 环境）：
    python src/simulation/scripts/train_lift.py \
        [--num-envs 1024] [--max-iterations 3000] [--log-dir outputs/rl/lift_cup]
"""

import argparse
import sys
from pathlib import Path

_SIM_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _SIM_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))

# 注意顺序：SimContext 必须在任何 simulation.tasks / isaaclab 导入之前进入
from simulation.sim_context import SimContext  # noqa: E402

TASK_ID = "Saw-LiftCup-FrankaDual-v0"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-envs", type=int, default=None,
                    help="并行环境数，默认取 simulation.yaml env.rl_num_envs")
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--log-dir", type=Path,
                    default=Path("outputs/rl/lift_cup"))
    ap.add_argument("--seed", type=int, default=2)
    #ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--headless", action="store_true",
                    help="无头模式（2026-09-23 用户裁决：默认开可视化窗口，"
                         "加此 flag 才无头——远程/批量跑时用）")
    ap.add_argument("--debug-cup", action="store_true",
                    help="在杯几何中心（reach 目标点）画 RGB 坐标轴"
                         "（lift_env_cfg.cup_center_frame 的 debug_vis）")
    ap.add_argument("--resume", type=Path, default=None,
                    help="从 ckpt 续训（.pt 路径）：恢复权重+优化器+迭代计数，"
                         "rsl_rl learn() 从断点轮次继续（2026-09-21 iter434 "
                         "崩溃后续训需求，接 model_400.pt）")
    args = ap.parse_args()

    with SimContext(headless=args.headless, enable_cameras=False) as ctx:
        import simulation.tasks  # noqa: F401  （触发 gym.register）
        from simulation.rl import facade
        from simulation.tasks.hang_cup.lift_env_cfg import make_rl_cfg
        from simulation.tasks.hang_cup.agents.rsl_rl_ppo_cfg import (
            HangCupPPORunnerCfg)

        cfg = make_rl_cfg(args.num_envs)
        if args.debug_cup:  # 场景尚未实例化，直接改 cfg 生效
            cfg.scene.cup_center_frame.debug_vis = True
        env = ctx.make_env(TASK_ID, cfg=cfg)
        print(f"env: {TASK_ID}, num_envs={cfg.scene.num_envs}, "
              f"obs policy dim="
              f"{env.unwrapped.observation_manager.group_obs_dim['policy']}")

        wrapped = facade.wrap_env_for_training(env)
        agent_cfg = HangCupPPORunnerCfg()
        # 网络只输出左手 8 维（2026-09-23 用户裁决"右手不参与"）：
        # 动作项是 8 维 LeftArmDeltaBinaryGripperAction（右臂每步锁
        # home + 全开），rsl_rl 按 env.num_actions=8 自动建 8 维
        # actor——不需要自定义策略类。
        if args.max_iterations is not None:
            agent_cfg.max_iterations = args.max_iterations
        if args.seed is not None:
            agent_cfg.seed = args.seed
        args.log_dir.mkdir(parents=True, exist_ok=True)
        runner = facade.make_runner(wrapped, agent_cfg,
                                    log_dir=str(args.log_dir))
        if args.resume is not None:
            runner.load(str(args.resume))   # 权重+优化器+iter 计数全恢复
            print(f"续训: {args.resume} → iter {runner.current_learning_iteration}")
        runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                     init_at_random_ep_len=True)
        print(f"-> {args.log_dir}（策略 ckpt 在各 run 目录的 model.pt）")


if __name__ == "__main__":
    main()
