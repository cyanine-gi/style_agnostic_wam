#!/usr/bin/env python
"""hang_cup 可视化检查器：GUI 实时查看仿真状态 + 依赖链各级门控诊断打印。

用途（2026-09-18 需求）：肉眼核对场景/策略行为，同时盯奖励依赖链每一级
的物理量与门控布尔——指尖距杯、指-杯接触力、held、准静态、离地、
目标距离、hung 判据——确认"前级没满足时后级恒 0"。

三种动作来源：
    默认          零增量（delta 契约下 = 保持 home 姿态不动），纯看场景
    --checkpoint  加载 RL ckpt 跑策略 rollout（目录含 model.pt 或直接 .pt）
    --random      小幅随机增量（物理响应冒烟）

用法（env_isaaclab 环境）：
    python src/simulation/scripts/view_hang_cup.py
    python src/simulation/scripts/view_hang_cup.py --checkpoint outputs/rl/hang_cup_v8/<run>
    python src/simulation/scripts/view_hang_cup.py --random --speed 2.0

诊断行每 --print-every 步打印一次（默认 15 步 = 1s @ 15Hz）：
    d_tip   右指尖中点到杯心距离（c0 瞄准判据 <5cm 且张爪）
    F_R     右手指-杯接触力（c1/c2 判据）
    F_cuptab 杯-桌接触力（c3 离地判据 <1N）
    d_slot/z_rel  杯相对钉的水平距离/杯底相对钉顶高度（c4/c5 判据）
    状态    z=z_latch 锁存阶段(0-7)、c=谓词真值串 c0..c6、clampR=_clamp、
            succ_run=成功持续计数（30 帧点火）、SUCCESS=成功终止标志
关闭窗口或 Ctrl+C 退出。
"""

import argparse
import sys
import time
from pathlib import Path

_SIM_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _SIM_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))

# SimContext 必须先于任何 simulation.tasks / isaaclab 导入进入
from simulation.sim_context import SimContext  # noqa: E402

_REST_CUP_Z = 0.777   # 静止杯心高度（桌面 0.75 + 杯半高 0.027，杯 ×0.6 后）


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="RL ckpt（目录或 .pt）；不给则零增量保持 home")
    ap.add_argument("--task", choices=["hang", "lift"], default="hang",
                    help="hang=挂杯状态机任务（默认）；lift=原生举起任务"
                         "（Saw-LiftCup-FrankaDual-v0，诊断列相应切换）")
    ap.add_argument("--random", action="store_true",
                    help="小幅随机增量动作（与 --checkpoint 互斥）")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="相对真实时间的倍速（1.0=约实时，15Hz 控制）")
    ap.add_argument("--print-every", type=int, default=15,
                    help="诊断行打印间隔（步）")
    args = ap.parse_args()
    assert not (args.checkpoint and args.random), "--checkpoint 与 --random 互斥"

    with SimContext(headless=False, enable_cameras=True) as ctx:
        import torch

        import simulation.tasks  # noqa: F401  （触发 gym.register）
        from simulation.rl import facade
        from simulation.tasks.hang_cup import mdp, mdp_lift
        from simulation.tasks.hang_cup.env_cfg import load_sim_config
        from simulation.tasks.hang_cup.agents.rsl_rl_ppo_cfg import (
            HangCupPPORunnerCfg)

        task_id = ("Saw-HangCup-FrankaDual-v0" if args.task == "hang"
                   else "Saw-LiftCup-FrankaDual-v0")
        mdp_sm = None
        if args.task == "hang":
            from simulation.tasks.hang_cup import mdp_sm as _mdp_sm
            mdp_sm = _mdp_sm
        env = ctx.make_env(task_id, num_envs=1)
        u = env.unwrapped

        # 视角：斜视工位全景（近似 front 录制机位，可在 GUI 里鼠标调整）
        try:
            u.sim.set_camera_view([0.0, -1.3, 1.6], [0.0, 0.0, 0.8])
        except Exception as e:  # 视角设置失败不影响主功能
            print(f"[warn] 视角设置失败（可手动导航）: {e}")

        policy = None
        wrapped = None
        if args.checkpoint:
            wrapped = facade.wrap_env_for_training(env)
            policy = facade.load_policy(str(args.checkpoint), wrapped,
                                        HangCupPPORunnerCfg())
            print(f"策略: {args.checkpoint}")
        elif args.random:
            print("动作来源: 随机增量 (0.3×randn)")
        else:
            print("动作来源: 零增量（保持 home 姿态）")

        rng = torch.Generator(device=u.device).manual_seed(0)
        zeros = torch.zeros(1, u.action_manager.total_action_dim, device=u.device)

        def diagnostics():
            cup = u.scene["cup"].data.root_pos_w - u.scene.env_origins
            vel = torch.linalg.norm(
                u.scene["cup"].data.root_lin_vel_w[0]).item()
            d_tip = torch.linalg.norm(
                mdp._tip_mid(u, "right") - cup, dim=1)[0].item()
            f_r = mdp._finger_cup_force(u, "right")[0].item()
            grip = u.action_manager.get_term("arm").processed_actions[0]
            grip_open = grip[15].item()          # 右爪开口（m，0.04=全张）
            h = cup[0, 2].item() - mdp_lift.CUP_HALF_H - mdp_lift.TABLE_Z
            if mdp_sm is None:
                # lift 任务：盯 d_tip（reach）、F_R（接触）、h（举起主项）
                print(f"  d_tip={d_tip*100:5.1f}cm  F_R={f_r:4.2f}N"
                      f"  grip={grip_open*100:4.1f}cm  |v|={vel:4.2f}m/s"
                      f"  杯底离桌 h={h*100:+5.1f}cm（成功线 15.0）")
                return
            # 2026-09-21 路线 A：诊断行改盯状态机——谓词真值串 c0..c6、
            # z_latch、success 计数；物理量保留（指尖距/接触力/桌力/杯速）。
            st = mdp_sm.update_stage_machine(u)
            pred = "".join("1" if b else "0" for b in st.true[0].tolist())
            d_slot = torch.linalg.norm(
                cup[0, :2] - st._slot_xy[0]).item() if st._slot_xy is not None \
                else float("nan")
            z_rel = (cup[0, 2] - mdp_sm.SM_PARAMS.cup_half_h
                     - (st._z_top[0] if st._z_top is not None
                        else torch.tensor(float("nan")))).item()
            f_tab = mdp._sensor_force_sum(u, "contact_cup_table")[0].item()
            f_body = (mdp._sensor_force_sum(u, "contact_body_left")
                      + mdp._sensor_force_sum(u, "contact_body_right"))[0].item()
            clamp_r, min_f = mdp._clamp(u, "right")
            print(f"  d_tip={d_tip*100:5.1f}cm  F_R={f_r:4.2f}N"
                  f"  F_cuptab={f_tab:4.1f}N F_body={f_body:4.1f}N"
                  f"  grip={grip_open*100:4.1f}cm  |v|={vel:4.2f}m/s"
                  f"  cup_z={cup[0,2].item():.3f}"
                  f"  d_slot={d_slot*100:5.1f}cm z_rel={z_rel*100:+5.1f}cm"
                  f"  | z={int(st.z_latch[0])} c={pred}"
                  f"  clampR={int(clamp_r[0])}({min_f[0]:.1f}N)"
                  f"  succ_run={int(st.success_run[0])}"
                  f"  SUCCESS={int(st.success_flag[0])}")

        if wrapped is not None:
            obs, _ = wrapped.reset()
        else:
            env.reset()

        step_dt = 1.0 / 15.0 / max(args.speed, 1e-3)
        ep, t = 0, 0
        print("开始（关闭窗口或 Ctrl+C 退出）。诊断列含义见脚本 docstring。")
        try:
            while SimContext._app.is_running():
                t0 = time.perf_counter()
                if policy is not None:
                    with torch.no_grad():
                        action = policy(obs)
                    obs, _, dones, _ = wrapped.step(action)
                    done = bool(dones[0].item())
                else:
                    a = (0.3 * torch.randn(zeros.shape, generator=rng,
                                           device=u.device)
                         if args.random else zeros)
                    _, _, term, trunc, _ = env.step(a)
                    done = bool(term.any().item() or trunc.any().item())

                if t % args.print_every == 0:
                    diagnostics()
                t += 1

                if done:
                    ep += 1
                    print(f"[episode {ep} 结束 @ {t} 步] 重置")
                    if wrapped is not None:
                        obs, _ = wrapped.reset()
                    else:
                        env.reset()
                    t = 0

                # 约实时节流（GUI 模式默认全速跑会快放，看不清）
                remain = step_dt - (time.perf_counter() - t0)
                if remain > 0:
                    time.sleep(remain)
        except KeyboardInterrupt:
            print("退出。")


if __name__ == "__main__":
    main()
