#!/usr/bin/env python
"""hang_cup 奖励逐项诊断：跑策略 rollout，逐项打印/统计奖励谁在付钱。

2026-09-21 路线 A 状态机版：奖励项变为 first_visit/shape/time/safety 等，
诊断重心从"谁在持续付钱"转为"状态机推进到哪、谓词原始值卡在哪"。
门控物理量打印换成：z_latch、谓词真值串 c0..c6、右臂 _clamp、指-杯力。
汇总表保留三桶口径（全程/离地/离地未夹持）——"离地未夹持"列非零的
整形项仍是可疑项（状态机版理论上只有 shape 项可能非零且应为负）。

用法（env_isaaclab 环境）：
    python src/simulation/scripts/diag_reward_terms.py \
        --checkpoint outputs/rl/hang_cup_xxx/model_1200.pt
    python src/simulation/scripts/diag_reward_terms.py --episodes 8 --quiet
"""

import argparse
import sys
from pathlib import Path

_SIM_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _SIM_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))

# SimContext 必须先于任何 simulation.tasks / isaaclab 导入进入
from simulation.sim_context import SimContext  # noqa: E402

_REST_CUP_Z = 0.777   # 静止杯心高度（桌面 0.75 + 杯半高 0.027，杯 ×0.6 后）


def _reward_terms(env):
    """从 RewardManager 取 (名称, cfg) 列表（逐版本兼容）。"""
    rm = env.unwrapped.reward_manager
    if hasattr(rm, "_term_names") and hasattr(rm, "_term_cfgs"):
        return list(zip(rm._term_names, rm._term_cfgs))
    from isaaclab.managers import RewardTermCfg
    return [(k, v) for k, v in vars(rm.cfg).items()
            if isinstance(v, RewardTermCfg)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("outputs/rl/hang_cup_v14/model_1200.pt"),
                    help="RL ckpt（.pt 或含 model.pt 的目录）")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=450,
                    help="单 episode 步数上限（默认 > episode_length，跑满自然终止）")
    ap.add_argument("--print-every", type=int, default=15,
                    help="诊断行打印间隔（步）；离地/夹持步强制打印")
    ap.add_argument("--quiet", action="store_true", help="只看尾声汇总表")
    args = ap.parse_args()

    with SimContext(headless=True, enable_cameras=False) as ctx:
        import torch

        import simulation.tasks  # noqa: F401  （触发 gym.register）
        from simulation.rl import facade
        from simulation.tasks.hang_cup import mdp, mdp_sm
        from simulation.tasks.hang_cup.env_cfg import make_rl_cfg, load_sim_config
        from simulation.tasks.hang_cup.agents.rsl_rl_ppo_cfg import (
            HangCupPPORunnerCfg)

        cfg = make_rl_cfg(num_envs=1)
        env = ctx.make_env(load_sim_config()["task_id"], cfg=cfg)
        u = env.unwrapped
        wrapped = facade.wrap_env_for_training(env)
        policy = facade.load_policy(str(args.checkpoint), wrapped,
                                    HangCupPPORunnerCfg())
        print(f"策略: {args.checkpoint}")

        terms = _reward_terms(env)
        dt = 1.0 / 15.0
        print(f"奖励项: {[n for n, _ in terms]}")

        # 汇总桶：{term: [全程, airborne, airborne&!held]}
        bucket = {n: [0.0, 0.0, 0.0] for n, _ in terms}
        n_steps = [0, 0, 0]

        obs, _ = wrapped.reset()
        for ep in range(args.episodes):
            for t in range(args.max_steps):
                with torch.no_grad():
                    action = policy(obs)
                obs, _, dones, _ = wrapped.step(action)

                cup_z = (u.scene["cup"].data.root_pos_w[0, 2]
                         - u.scene.env_origins[0, 2]).item()
                # 状态机版的 held = 右臂 _clamp（谓词 c2 的核心判据）
                c_r, m_r = mdp._clamp(u, "right")
                held = bool(c_r[0].item())
                st = mdp_sm.update_stage_machine(u)
                airborne = cup_z - _REST_CUP_Z > 0.03
                n_steps[0] += 1
                n_steps[1] += int(airborne)
                n_steps[2] += int(airborne and not held)

                vals = {}
                for name, tc in terms:
                    v = float(tc.func(u, **tc.params)[0].item()) * tc.weight
                    vals[name] = v
                    bucket[name][0] += v * dt
                    if airborne:
                        bucket[name][1] += v * dt
                        if not held:
                            bucket[name][2] += v * dt

                interesting = airborne or held
                if not args.quiet and (t % args.print_every == 0 or interesting):
                    d_tip = torch.linalg.norm(
                        mdp._tip_mid(u, "right")
                        - (u.scene["cup"].data.root_pos_w
                           - u.scene.env_origins), dim=1)[0].item()
                    grip = u.action_manager.get_term("arm").processed_actions[0]
                    pred = "".join("1" if b else "0"
                                   for b in st.true[0].tolist())
                    nz = "  ".join(f"{k}={v:+.2f}" for k, v in vals.items()
                                   if abs(v) > 1e-3)
                    print(f"ep{ep} t={t:3d} cup_z={cup_z:.3f} "
                          f"d_tip={d_tip*100:4.1f}cm "
                          f"clampR={int(c_r[0])}({m_r[0]:.1f}N) "
                          f"grip={grip[15].item()*100:3.1f}cm "
                          f"z={int(st.z_latch[0])} c={pred} "
                          f"held={int(held)} air={int(airborne)} | {nz}")

                if bool(dones[0].item()):
                    print(f"[episode {ep} 结束 @ {t} 步]")
                    break
            obs, _ = wrapped.reset()

        print("\n===== 奖励逐项累计（与 PPO 实吃口径一致 = Σ value×weight×dt）=====")
        print(f"步数: 全程 {n_steps[0]} | 离地 {n_steps[1]} | "
              f"离地未夹持 {n_steps[2]}")
        print(f"{'term':<16}{'全程':>10}{'离地':>10}{'离地未夹持':>12}")
        for name, _ in terms:
            a, b, c = bucket[name]
            print(f"{name:<16}{a:>10.2f}{b:>10.2f}{c:>12.2f}")
        print("\n判读：'离地未夹持' 列非零 = 没夹住时也在拿钱；状态机版里"
              "该列只有 shape 可能非零（负值引导，无正向营地），"
              "first_visit/success_bonus 该列应恒 0（锁存推进序保证"
              "离地必先过 c2 夹持）。")


if __name__ == "__main__":
    main()
