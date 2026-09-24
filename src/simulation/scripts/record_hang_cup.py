#!/usr/bin/env python
"""hang_cup 数据集录制：RL 策略驱动 + 每集相机随机化 + RoboMIND 子集落盘。

2026-09-17 sawwam 裁决（src/sawwam/guideline.md §6.3 硬前提）：
- 每集 reset 后按 YAML randomize 块独立随机化各路相机（pos/look_at/
  focal），集内固定；精确 K/T 逐集写进 HDF5（约定见 recording.py 文档）；
- 动作来源 = 训练好的 RL 策略（train_rl.py 产物）；只落盘挂杯成功的
  episode（success_episodes 目录语义），--keep-failed 可存失败集。
- 帧对齐：每步先读帧/本体感（时刻 t 的状态），再记策略动作（时刻 t 的
  指令），最后 step——(rgb_t, puppet_t, master_t) 三者同刻，对齐
  RoboMIND 的 align 曲线语义。

用法（需要 env_isaaclab 环境）：
    python src/simulation/scripts/record_hang_cup.py \
        --checkpoint outputs/rl/hang_cup/<run_dir> --num-episodes 300
冒烟（先验证帧对齐与 K/T round-trip，见输出打印）：
    python src/simulation/scripts/record_hang_cup.py \
        --checkpoint <run_dir> --num-episodes 3 --outdir /tmp/rec_smoke
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
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="runner log 目录（内含 model.pt）或直接给 .pt 文件")
    ap.add_argument("--num-episodes", type=int, default=300)
    ap.add_argument("--outdir", type=Path,
                    default=Path("data/isaaclab_franka_dual"),
                    help="数据集根（内建 data/<robot>/<task>/... 布局）")
    ap.add_argument("--robot-dir", default="franka_isaaclab")
    ap.add_argument("--task-dir", default="hang_cup_on_rack")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-failed", action="store_true",
                    help="失败集也落盘到 failed_episodes/（默认丢弃）")
    args = ap.parse_args()

    import datetime
    import numpy as np

    with SimContext(headless=True, enable_cameras=True) as ctx:
        import simulation.tasks  # noqa: F401  （触发 gym.register）
        from simulation import recording as rec
        from simulation import viz
        from simulation.rl import facade
        from simulation.tasks.hang_cup import mdp_sm
        from simulation.tasks.hang_cup.env_cfg import (
            camera_names, load_sim_config)
        from simulation.tasks.hang_cup.agents.rsl_rl_ppo_cfg import (
            HangCupPPORunnerCfg)
        sim_cfg = load_sim_config()
        names = camera_names()
        env = ctx.make_env(sim_cfg["task_id"], num_envs=1)
        wrapped = facade.wrap_env_for_training(env)
        policy = facade.load_policy(str(args.checkpoint), wrapped,
                                    HangCupPPORunnerCfg())
        rng = np.random.default_rng(args.seed)
        cams = {n: env.unwrapped.scene.sensors[f"camera_{n}"]
                for n in names}
        max_steps = env.unwrapped.max_episode_length
        print(f"env: {sim_cfg['task_id']}, 相机 {names}, "
              f"每集上限 {max_steps} 步 @ {rec.CONTROL_HZ}Hz")

        n_saved = n_failed = 0
        for ep in range(args.num_episodes):
            obs, _ = wrapped.reset()

            # 每集：独立随机化各路相机（集内固定）
            poses = rec.sample_camera_poses(sim_cfg["cameras"], rng)
            rec.randomize_cameras(env, poses)
            obs, _, _, _ = wrapped.step(policy(obs))   # 空步强制渲染刷新

            # 基座系 + 精确 K/T（集内静态，只读一次）
            T_world_base = rec.compute_base_frame(env)
            KT = {n: rec.camera_extrinsics_in_base(cams[n], T_world_base)
                  for n in names}
            b2r = rec.robot_base_transforms(env, T_world_base)
            if ep == 0:
                print(f"基座系原点（应≈[0,0.15,0.75]）："
                      f"{np.round(T_world_base[:3, 3], 4)}")
                for n in names:
                    print(f"  {n}: K 请求 fx="
                          f"{rec.pinhole_K(cams[n].cfg.width, cams[n].cfg.height, poses[n]['focal_mm'])[0, 0]:.2f} "
                          f"实测 fx={KT[n][0][0, 0]:.2f}")

            writer = rec.EpisodeWriter(names)
            success = False
            for t in range(max_steps):
                # 先读时刻 t 的状态（帧 + 本体感），再取时刻 t 的指令
                rgb = {n: cams[n].data.output["rgb"][0, ..., :3]
                       .cpu().numpy() for n in names}
                depth = {n: viz.read_depth_mm(cams[n]).astype(np.uint16)
                         for n in names}
                puppet16 = rec.absolute_joint16(env)
                action = policy(obs)
                obs, _, dones, _ = wrapped.step(action)
                # 录制与动作空间解耦（2026-09-18 用户裁决）：落盘 master =
                # 动作项加工后的 16 维**绝对**目标；策略接口是 delta 增量，
                # 其 raw 输出不具数据契约语义，不可直接落盘。
                term = env.unwrapped.action_manager.get_term("arm")
                master16 = rec.flip_gripper16(
                    term.processed_actions[0].detach().cpu().numpy())
                writer.add_frame(rgb, depth, master16, puppet16,
                                 t * 1_000_000 // rec.CONTROL_HZ)
                if dones[0]:
                    # 2026-09-21 路线 A：成功 = 状态机 sm_success 终止
                    # （c5 插入持续 30 帧 + 准静态）。env 已在 step 内自动
                    # reset，live 谓词已清——sm_reset_state 在清理前把结果
                    # 快照进 last_episode_success，读它。
                    success = bool(mdp_sm._state(
                        env.unwrapped).last_episode_success[0].item())
                    break

            stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            ep_name = f"isaac-{stamp}-{ep:04d}"
            if success:
                out = (args.outdir / "data" / args.robot_dir / args.task_dir
                       / "success_episodes" / ep_name / "data"
                       / f"{ep_name}.hdf5")
                writer.save(out,
                            intrinsics={n: KT[n][0] for n in names},
                            extrinsics={n: KT[n][1] for n in names},
                            base_to_robot=b2r)
                n_saved += 1
                print(f"episode {ep}: 成功，{len(writer)} 帧 -> {out}")
            else:
                n_failed += 1
                if args.keep_failed:
                    out = (args.outdir / "data" / args.robot_dir
                           / args.task_dir / "failed_episodes" / ep_name
                           / "data" / f"{ep_name}.hdf5")
                    writer.save(out,
                                intrinsics={n: KT[n][0] for n in names},
                                extrinsics={n: KT[n][1] for n in names},
                                base_to_robot=b2r)
                print(f"episode {ep}: 失败（{len(writer)} 帧）"
                      + ("，已存 failed_episodes" if args.keep_failed else ""))

        print(f"完成：成功 {n_saved} / 失败 {n_failed} "
              f"（成功率 {n_saved / max(1, n_saved + n_failed):.1%}）")


if __name__ == "__main__":
    main()
