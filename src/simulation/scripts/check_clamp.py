#!/usr/bin/env python
"""夹持奖励链确定性验证 v3（2026-09-21 路线 A 状态机版）：**正常操作流程**，
零传送。物理操作序列与 v2 相同（IK 伺服：悬停 → 下降到杯心 → 闭拢 →
抬升），验证对象换成锁存阶段机（mdp_sm）：
  P1. 张爪移到杯子上方（杯顶 +12cm）；
  P2. 下降到杯心高度（指垫跨杯壁两侧）→ c0 瞄准应点火；
  A.  闭拢 45 步 → c1 接触 / c2 夹持（_clamp 三条件）应点火、
      z_latch 推进到 ≥2；
  B.  保持满压抬升 +15cm → c3 离地应点火、z_latch 推进到 ≥3，
      期间无 collision/drop 终止。

判读标准（沿用 v2 裁决的物理门槛 + 状态机推进）：
  - P1 指端误差 <1cm；
  - B 中 cup_z 上升 >3cm、无 term/trunc、c3 点火 ≥3 步。
  A 段 c2 是否立即点火不做要求（旧杯验证史：骑夹余量小易被杯沿楔成
  单指顶住，抬升受力后才入座成真实双指夹持）。

用法：
    python src/simulation/scripts/check_clamp.py [--gui] [--side left|right]
"""

import argparse
import sys
from pathlib import Path

_SIM_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _SIM_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))

# SimContext 必须先于任何 simulation.tasks / isaaclab 导入进入
from simulation.sim_context import SimContext  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gui", action="store_true", help="开 GUI 窗口（默认 headless）")
    ap.add_argument("--side", choices=["left", "right"], default="right",
                    help="用哪只臂验证。2026-09-21 起状态机谓词只认右臂"
                         "（文档固定分工），选 left 时 z_latch/c0-c3 不会"
                         "推进，只剩物理夹持量可读")
    args = ap.parse_args()

    with SimContext(headless=not args.gui, enable_cameras=False) as ctx:
        import torch

        import simulation.tasks  # noqa: F401  （触发 gym.register）
        from isaaclab.controllers import DifferentialIKController
        from isaaclab.controllers.differential_ik_cfg import (
            DifferentialIKControllerCfg)
        from isaaclab.utils.math import (matrix_from_quat, quat_inv,
                                         subtract_frame_transforms)
        from simulation.tasks.hang_cup import mdp, mdp_sm
        from simulation.tasks.hang_cup.env_cfg import (
            load_sim_config, make_rl_cfg)

        sim_cfg = load_sim_config()
        env = ctx.make_env(sim_cfg["task_id"], cfg=make_rl_cfg(num_envs=1))
        u = env.unwrapped
        dev = u.device
        side = args.side
        g_dim = 14 if side == "left" else 15      # [L7,R7,gL,gR] 契约
        arm_slice = slice(0, 7) if side == "left" else slice(7, 14)

        robot = u.scene[f"robot_{side}"]
        arm_ids, _ = robot.find_joints(
            [f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)
        hand_id = robot.find_bodies("panda_hand")[0][0]
        assert robot.is_fixed_base, "jacobian 索引约定仅适用于固定基座"

        # 差分 IK：绝对位姿命令、基座系（frame 约定照抄 2.3.2
        # task_space_actions.py:230-256 / differential_ik.py:99-185）
        ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose",
                                        use_relative_mode=False,
                                        ik_method="dls"),
            num_envs=1, device=dev)
        ik.reset()

        origins = u.scene.env_origins                     # (1,3)

        def ee_pose_b():
            return subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w,
                robot.data.body_pos_w[:, hand_id],
                robot.data.body_quat_w[:, hand_id])

        def jacobian_b():
            J = robot.root_physx_view.get_jacobians()[
                :, hand_id - 1, :, arm_ids].clone()       # 固定基座: 索引-1
            R = matrix_from_quat(quat_inv(robot.data.root_quat_w))
            J[:, :3, :] = torch.bmm(R, J[:, :3, :])
            J[:, 3:, :] = torch.bmm(R, J[:, 3:, :])
            return J

        def tip_mid_w():
            return mdp._finger_positions(u, side)[0].mean(dim=0) + origins[0]

        def w2b(p_w):
            """世界系点 → 该臂基座系。"""
            return torch.bmm(
                matrix_from_quat(quat_inv(robot.data.root_quat_w)),
                (p_w - robot.data.root_pos_w).unsqueeze(-1)).squeeze(-1)

        zeros = torch.zeros(1, 16, device=dev)

        def ik_step(tgt_w, grip_raw):
            """一步 IK 伺服 + 夹爪 raw，返回指端中点到目标的误差(m)。

            坐标链（2026-09-19 第二版修正）：IK 控的是 **hand 原点**
            （jacobian/ee_pose 都是 panda_hand 系），而任务目标点是
            **指端中点**——两者差 ~10cm（手指自 hand 向下延伸）。
            由于全程锁定同一朝向（hold_quat_b），hand→指端偏置在基座系
            是常量（settle 后量取），命令时补偿：cmd = w2b(tgt) + off。
            """
            ep, eq = ee_pose_b()
            q_cur = robot.data.joint_pos[:, arm_ids]
            cmd_pos = w2b(tgt_w) + tip_off_b
            ik.set_command(torch.cat([cmd_pos, hold_quat_b], dim=-1))
            q_des = ik.compute(ep, eq, jacobian_b(), q_cur)
            a = zeros.clone()
            a[0, arm_slice] = ((q_des - q_cur) / 0.1).clamp(-1.0, 1.0)[0]
            a[0, g_dim] = grip_raw
            env.step(a)
            return torch.linalg.norm(tip_mid_w() - tgt_w).item()

        def report(step, phase):
            V = mdp._finger_cup_force_vec(u, side)[0]          # (2,3)
            f = torch.linalg.norm(V, dim=-1)
            vn = V / f.clamp_min(1e-6)[:, None]
            cos = (vn[0] * vn[1]).sum().item()
            clamp, min_f = mdp._clamp(u, side)
            st = mdp_sm.update_stage_machine(u)
            pred = "".join("1" if b else "0" for b in st.true[0].tolist())
            cup_z = (u.scene["cup"].data.root_pos_w
                     - u.scene.env_origins)[0, 2].item()
            term = u.action_manager.get_term("arm")
            grip = term.processed_actions[0, g_dim].item()
            cur = term._current16()[0, g_dim].item()   # 实测夹爪关节（两指均值）
            f_body = mdp._sensor_force_sum(u, f"contact_body_{side}")[0].item()
            print(f"  [{phase} {step:3d}]"
                  f" F1=({V[0,0]:+5.1f},{V[0,1]:+5.1f},{V[0,2]:+5.1f})|{f[0]:4.1f}N"
                  f" F2=({V[1,0]:+5.1f},{V[1,1]:+5.1f},{V[1,2]:+5.1f})|{f[1]:4.1f}N"
                  f" Fbody={f_body:4.1f}N"
                  f" cos={cos:+5.2f} clamp={int(clamp[0])} min_f={min_f[0]:5.2f}"
                  f" | z_latch={int(st.z_latch[0])} c={pred}"
                  f" R1st={st._latch_delta[0]:.0f}"
                  f" succ_run={int(st.success_run[0])}"
                  f" pen_fa={int(mdp.pen_finger_arm(u)[0])}"
                  f" | g={grip*100:4.1f}/{cur*100:4.1f}cm(指令/实测)"
                  f" cup_z={cup_z:.3f}")

        env.reset()
        for _ in range(20):                    # home 稳定
            env.step(zeros)

        # GUI 初始视角：平视杯子，从操作臂的反方向看过去
        # （相机在"杯−臂基座"水平连线的延长线上、与杯同高，2026-09-19 用户定）
        view_eye = view_tgt = None
        if args.gui:
            cw = u.scene["cup"].data.root_pos_w[0]
            bw = robot.data.root_pos_w[0]
            d = cw[:2] - bw[:2]
            d = d / torch.linalg.norm(d)          # 臂→杯 水平方向
            view_eye = [float(cw[0] + d[0] * 0.5),  # 相机在杯后 0.5m
                        float(cw[1] + d[1] * 0.5),
                        float(cw[2])]               # 平视：与杯同高
            view_tgt = [float(cw[0]), float(cw[1]), float(cw[2])]

        def apply_view():
            """重放视角（幂等）。Kit 在 stage 加载完成后会**异步**重放
            视口默认机位（2026-09-19 确诊：设置后闪一下又被盖回默认），
            所以在脚本前十几秒周期性重放，覆盖事件过去后即为终态。"""
            if view_eye is None:
                return
            try:
                u.sim.set_camera_view(view_eye, view_tgt)
            except Exception as e:
                print(f"[warn] 视角设置失败（可手动导航）: {e}")

        apply_view()

        # 操作目标点（世界系，杯全程不动——从 spawn 位姿读一次）
        hold_quat_b = ee_pose_b()[1].clone()   # 全程保持 home 手爪朝向
        tip_off_b = (ee_pose_b()[0] - w2b(tip_mid_w())).clone()  # hand→指端
        cup_w = u.scene["cup"].data.root_pos_w[0].clone()
        cup_z0 = (u.scene["cup"].data.root_pos_w
                  - u.scene.env_origins)[0, 2].item()
        hover = cup_w + torch.tensor([0.0, 0.0, 0.12], device=dev)
        grasp = cup_w.clone()                  # 杯心高度：指垫跨杯壁
        lift = grasp + torch.tensor([0.0, 0.0, 0.15], device=dev)

        def _fmt(p_w):
            p = p_w - origins[0]
            return f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})"

        print(f"杯位姿(env系): {_fmt(u.scene['cup'].data.root_pos_w[0])}"
              f"  指端中点: {_fmt(tip_mid_w())}"
              f"  hand→指端偏置(基座系): "
              f"({tip_off_b[0,0]:+.3f}, {tip_off_b[0,1]:+.3f}, {tip_off_b[0,2]:+.3f})")

        # ---- P1: 张爪悬停杯上方 ----------------------------------------
        err = 1e9
        for t in range(250):
            err = ik_step(hover, grip_raw=1.0)
            if t % 10 == 0:
                apply_view()               # 压制 Kit 异步默认机位重放
            if err < 0.01:
                break
        u.episode_length_buf.zero_()
        print(f"P1 悬停杯上方: {t+1} 步, 指端误差 {err*100:.1f}cm"
              f"  指端 {_fmt(tip_mid_w())} 目标 {_fmt(hover)}"
              + ("（未收敛!）" if err >= 0.01 else ""))

        # ---- P2: 下降到夹持位（接触感知：楔住即停降，不硬顶） -----------
        # 2026-09-19 确诊：无接触感知的下降在对准偏差时会把手掌/手指压在
        # 杯上持续硬顶（PD 力轴 + PhysX 柔顺接触 ⇒ 恒定的视觉穿插画面）。
        # 判则：离目标 >1.5cm 且 指-杯或身-杯力 >2N = 被杯挡住（楔住），
        # 停止下降而不是顶满 150 步；近目标（<1.5cm）的轻微接触属正常。
        # 2026-09-20：楔住**不再中止脚本**——旧杯验证史（run2）显示楔住的
        # 单指顶住态在闭拢/抬升受力后会重新入座成真实双指夹持（验收门槛
        # 本就不要求 A 段 clamp）；中止会挡住 A/B 段的奖励剖面观测。
        jammed = False
        f_fing = f_body = 0.0
        for t in range(150):
            err = ik_step(grasp, grip_raw=1.0)
            f_fing = mdp._finger_cup_force(u, side)[0].item()
            f_body = mdp._sensor_force_sum(u, f"contact_body_{side}")[0].item()
            if err > 0.015 and (f_fing > 2.0 or f_body > 2.0):
                jammed = True
                break
            if err < 0.005:
                break
        u.episode_length_buf.zero_()
        print(f"P2 下降到夹持位: {t+1} 步, 指端误差 {err*100:.1f}cm"
              f"  指端 {_fmt(tip_mid_w())} 目标 {_fmt(grasp)}"
              + (f"（楔住停降! 指-杯 {f_fing:.1f}N 身-杯 {f_body:.1f}N"
                 f" ——就地带楔闭爪，观察 A/B 段能否入座）" if jammed
                 else ("（未收敛!）" if err >= 0.005 else "")))

        # ---- A: 闭拢 45 步 ---------------------------------------------
        apply_view()
        print("A. 闭爪（期望 f1/f2>0.3N 且 cos<0.5 → clamp=1，"
              "c1/c2 谓词点火、z_latch→2）")
        for t in range(45):
            ik_step(grasp, grip_raw=-1.0)      # 满压闭拢（delta 语义：
            if t % 3 == 0:                     # 目标深度=夹持力）
                report(t, "A")
        u.episode_length_buf.zero_()
        clamp_end = bool(mdp._clamp(u, side)[0][0])
        print(f"  A 结束稳态 clamp={int(clamp_end)}")

        # ---- B: 保持满压，IK 抬升 +15cm ---------------------------------
        apply_view()
        print("B. 抬臂（期望 cup_z 升 >3cm 期间 c3 离地点火、z_latch→3）")
        c3_hit = False
        c3_steps = 0
        aborted = False
        for t in range(80):
            ik_step(lift, grip_raw=-1.0)
            term = u.termination_manager.dones
            trunc = u.episode_length_buf >= u.max_episode_length
            if bool(term.any() or trunc.any()):
                print(f"  !! B 段 {t} 步触发 episode 结束（term/trunc）——"
                      f"夹持在抬臂中丢失或碰撞终止，判定失败")
                aborted = True
                break
            if t % 10 == 0:
                report(t, "B")
            st = mdp_sm.update_stage_machine(u)
            if bool(st.true[0, 3]):
                c3_hit = True
                c3_steps += 1

        # ---- 汇总 -------------------------------------------------------
        cup_z_end = (u.scene["cup"].data.root_pos_w
                     - u.scene.env_origins)[0, 2].item()
        rise = cup_z_end - cup_z0
        latch_end = int(mdp_sm.update_stage_machine(u).z_latch[0])
        print("=" * 60)
        print(f"结果: 末态 cup_z={cup_z_end:.3f}（起点 {cup_z0:.3f}，"
              f"抬升 {rise:+.3f}m）  c3 点火={c3_hit}（{c3_steps} 步）"
              f"  z_latch={latch_end}  A 段 clamp={int(clamp_end)}(诊断用)")
        # 验收门槛（沿用 v2 裁决的物理口径 + 状态机推进）：A 段 clamp 不
        # 要求——c3 离地点火本身 = "空中真实夹持"的物理证据（c3 需先过
        # c2 锁存推进）；要求 ≥3 步防抖动。
        ok = (not aborted) and c3_steps >= 3 and rise > 0.03
        print("判定:", "通过 ✓ 奖励链在真实操作分布下健康" if ok
              else "未通过 ✗（把上面的打印贴给我）")


if __name__ == "__main__":
    main()
