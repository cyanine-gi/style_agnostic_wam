#!/usr/bin/env python
"""hang_cup 状态机验收测试（路线 A 文档 §9，2026-09-21）。

七条验收闸——**测试 3 不过不进训练**（文档硬规定）：
  1. 脚本走理想轨迹（瞄准→接触→夹→抬→运→插→放），z_latch 单调 0→7，
     允许相邻阶段间重复，不允许无恢复的回跳；
  2. 抓杯后松手：z_latch 不回退、不动，不终止不罚分；
  3. 反复「夹起—放下」10 次：z_latch 不回退，累计 R_first 恒定；
  4. hack 探针各 50 步：空手悬停杯上（c0 不得持续为真）；空夹满力
     （c2 不得为真）；横扫扇飞杯子（success 不得触发）；
  5. 杯在 d_aim_enter 边界 ±1mm 抖动：c0 计数器不得高频翻转；
  6. 随机 reset ×100：首步全部谓词为假、z_latch=0；
  7. max_steps 超时：time_out 与 terminated 分离。

另附 --eval 模式（文档 §8）：加载 ckpt 跑 ≥100 episode，报成功率 +
部分分（max z_latch 均值 / 7）。

IK 伺服机器复用 check_clamp.py 的 DLS 差分 IK 模式（绝对位姿命令、
基座系、hand→指端偏置补偿、全程锁定 home 朝向）。状态机谓词只认右臂
（文档固定分工），全程操作右臂。

用法（env_isaaclab 环境）：
    python src/simulation/scripts/test_state_machine.py            # 测试 1-7
    python src/simulation/scripts/test_state_machine.py --only 3   # 只跑测试 3
    python src/simulation/scripts/test_state_machine.py \
        --eval outputs/rl/hang_cup_xxx/model_xxx.pt --episodes 100
"""

import argparse
import sys
from pathlib import Path

_SIM_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _SIM_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))

# SimContext 必须先于任何 simulation.tasks / isaaclab 导入进入
from simulation.sim_context import SimContext  # noqa: E402


# --------------------------------------------------------------------------- #
# IK 伺服（模式照抄 check_clamp.py：DLS、绝对位姿、基座系、指端偏置补偿）
# --------------------------------------------------------------------------- #

class ArmServo:
    def __init__(self, env, side="right"):
        import torch
        from isaaclab.controllers import DifferentialIKController
        from isaaclab.controllers.differential_ik_cfg import (
            DifferentialIKControllerCfg)

        self.env = env
        self.u = env.unwrapped
        self.dev = self.u.device
        self.side = side
        self.torch = torch
        self.g_dim = 14 if side == "left" else 15      # [L7,R7,gL,gR] 契约
        self.arm_slice = slice(0, 7) if side == "left" else slice(7, 14)

        self.robot = self.u.scene[f"robot_{side}"]
        self.arm_ids, _ = self.robot.find_joints(
            [f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)
        self.hand_id = self.robot.find_bodies("panda_hand")[0][0]
        assert self.robot.is_fixed_base

        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose",
                                        use_relative_mode=False,
                                        ik_method="dls"),
            num_envs=1, device=self.dev)
        self.ik.reset()
        self.zeros = torch.zeros(1, 16, device=self.dev)
        self.hold_quat_b = None
        self.tip_off_b = None

    def calibrate(self, mdp):
        """settle 后量取 home 朝向与 hand→指端偏置（基座系常量）。"""
        self.hold_quat_b = self.ee_pose_b()[1].clone()
        self.tip_off_b = (self.ee_pose_b()[0]
                          - self.w2b(self.tip_mid_w(mdp))).clone()

    def ee_pose_b(self):
        from isaaclab.utils.math import subtract_frame_transforms
        return subtract_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w,
            self.robot.data.body_pos_w[:, self.hand_id],
            self.robot.data.body_quat_w[:, self.hand_id])

    def jacobian_b(self):
        torch = self.torch
        from isaaclab.utils.math import matrix_from_quat, quat_inv
        J = self.robot.root_physx_view.get_jacobians()[
            :, self.hand_id - 1, :, self.arm_ids].clone()
        R = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        J[:, :3, :] = torch.bmm(R, J[:, :3, :])
        J[:, 3:, :] = torch.bmm(R, J[:, 3:, :])
        return J

    def tip_mid_w(self, mdp):
        return mdp._finger_positions(self.u, self.side)[0].mean(dim=0) \
            + self.u.scene.env_origins[0]

    def w2b(self, p_w):
        torch = self.torch
        from isaaclab.utils.math import matrix_from_quat, quat_inv
        return torch.bmm(
            matrix_from_quat(quat_inv(self.robot.data.root_quat_w)),
            (p_w - self.robot.data.root_pos_w).unsqueeze(-1)).squeeze(-1)

    def step(self, tgt_w, grip_raw, mdp):
        """一步 IK 伺服 + 夹爪 raw，返回 (指端误差 m, done)。"""
        torch = self.torch
        ep, eq = self.ee_pose_b()
        q_cur = self.robot.data.joint_pos[:, self.arm_ids]
        cmd_pos = self.w2b(tgt_w) + self.tip_off_b
        self.ik.set_command(torch.cat([cmd_pos, self.hold_quat_b], dim=-1))
        q_des = self.ik.compute(ep, eq, self.jacobian_b(), q_cur)
        a = self.zeros.clone()
        a[0, self.arm_slice] = ((q_des - q_cur) / 0.1).clamp(-1.0, 1.0)[0]
        a[0, self.g_dim] = grip_raw
        _, _, terminated, time_outs, _ = self.env.step(a)
        err = torch.linalg.norm(self.tip_mid_w(mdp) - tgt_w).item()
        done = bool((terminated[0] | time_outs[0]).item())
        return err, done

    def goto(self, tgt_w, grip_raw, mdp, max_steps=250, tol=0.01,
             jam_guard=True):
        """伺服到目标点；楔住（力>2N 且误差>1.5cm）提前停。返回
        (收敛?, done, 步数)。"""
        for t in range(max_steps):
            err, done = self.step(tgt_w, grip_raw, mdp)
            if done:
                return False, True, t + 1
            if err < tol:
                return True, False, t + 1
            if jam_guard and err > 0.015:
                f = mdp._finger_cup_force(self.u, self.side)[0].item()
                fb = mdp._sensor_force_sum(
                    self.u, f"contact_body_{self.side}")[0].item()
                if f > 2.0 or fb > 2.0:
                    return False, False, t + 1      # 楔住停降（不判败）
        return False, False, max_steps

    def goto_lerp(self, tgt_w, grip_raw, mdp, total_steps=150, tol=0.01,
                  jam_guard=True, hook=None):
        """目标点从当前指端位置线性插值缓动（探针确诊 2026-09-21：裸 IK
        绝对命令单步跳变厘米级，下降会单指插进杯壁楔死 189N；插值把
        轨迹限速到 ~1mm/步）。hook(step_ret) 每步回调（记录 latch 轨迹用）。
        返回 (收敛?, done, 步数)。"""
        start = self.tip_mid_w(mdp).clone()
        for t in range(1, total_steps + 1):
            tgt = start + (tgt_w - start) * (t / total_steps)
            err, done = self.step(tgt, grip_raw, mdp)
            if hook is not None:
                hook()
            if done:
                return False, True, t
            if t == total_steps and err < tol:
                return True, False, t
            if jam_guard and t > total_steps // 2:
                f = mdp._finger_cup_force(self.u, self.side)[0].item()
                fb = mdp._sensor_force_sum(
                    self.u, f"contact_body_{self.side}")[0].item()
                if f > 10.0 or fb > 10.0:
                    return False, False, t          # 硬楔停降（不判败）
        err = self.torch.linalg.norm(self.tip_mid_w(mdp) - tgt_w).item()
        return err < tol, False, total_steps

    def keep_distance_buf_fresh(self):
        """测试分段之间清零 episode 计数防超时（check_clamp 同款技巧）。"""
        self.u.episode_length_buf.zero_()


# --------------------------------------------------------------------------- #
# 测试主体
# --------------------------------------------------------------------------- #

def _sm(env, mdp_sm):
    return mdp_sm.update_stage_machine(env.unwrapped)


def _cup(env):
    return env.unwrapped.scene["cup"].data.root_pos_w \
        - env.unwrapped.scene.env_origins


def run_tests(ctx, args):
    import torch

    import simulation.tasks  # noqa: F401  （触发 gym.register）
    from simulation.tasks.hang_cup import mdp, mdp_sm
    from simulation.tasks.hang_cup.env_cfg import load_sim_config, make_rl_cfg

    sim_cfg = load_sim_config()
    env = ctx.make_env(sim_cfg["task_id"], cfg=make_rl_cfg(num_envs=1))
    u = env.unwrapped
    p = mdp_sm.SM_PARAMS
    results = {}

    def fresh_start():
        env.reset()
        servo = ArmServo(env, "right")
        z = servo.zeros
        for _ in range(20):
            env.step(z)
        servo.calibrate(mdp)
        servo.keep_distance_buf_fresh()
        return servo

    def done_or_trunc():
        term = u.termination_manager.dones
        trunc = u.episode_length_buf >= u.max_episode_length
        return bool(term.any() or trunc.any())

    # ------------------------------------------------------------------ #
    # 测试 1：理想轨迹 z_latch 单调 0→7
    # ------------------------------------------------------------------ #
    if args.only in (0, 1):
        print("== 测试 1：理想轨迹全程 ==")
        servo = fresh_start()
        latch_hist = []
        cup_w = _cup(env)[0].clone()
        peg_w = u.scene["peg"].data.root_pos_w[0].clone()

        # 瞄准 + 接触 + 夹持（下降用插值缓动：裸 IK 跳变会单指插壁楔死）
        hover = cup_w + torch.tensor([0.0, 0.0, 0.12], device=u.device)
        servo.goto(hover, 1.0, mdp)
        servo.keep_distance_buf_fresh()
        conv, done, _ = servo.goto_lerp(cup_w, 1.0, mdp, total_steps=150,
                                        tol=0.005)
        servo.keep_distance_buf_fresh()
        print(f"  下降到杯心: conv={conv} done={done}"
              f" d_tip={torch.linalg.norm(servo.tip_mid_w(mdp) - cup_w).item()*100:.1f}cm")
        for _ in range(60):                     # 闭拢（满压）
            _, done = servo.step(cup_w, -1.0, mdp)
            latch_hist.append(int(_sm(env, mdp_sm).z_latch[0]))
            if done:
                break
        servo.keep_distance_buf_fresh()
        print(f"  夹持后 z_latch={latch_hist[-1]}"
              f"（期望 ≥3：c0/c1/c2 已锁存；c3 需抬升）")

        # 抬升 → c3
        lift = cup_w + torch.tensor([0.0, 0.0, 0.18], device=u.device)
        for _ in range(80):
            _, done = servo.step(lift, -1.0, mdp)
            latch_hist.append(int(_sm(env, mdp_sm).z_latch[0]))
            if done or latch_hist[-1] >= 4:
                break
        servo.keep_distance_buf_fresh()

        # 运输到钉上方 → c4（指端平移 = 杯位移，杯爪近似刚性；插值缓动）
        st = _sm(env, mdp_sm)
        tip_now = servo.tip_mid_w(mdp)
        cup_now = _cup(env)[0]
        delta_xy = st._slot_xy[0] - cup_now[:2]
        over_slot = torch.cat([
            st._slot_xy[0],
            torch.tensor([st._z_top[0] + 0.10 + p.cup_half_h],
                         device=u.device)])              # 杯心目标
        tip_tgt = tip_now + (over_slot - cup_now)         # 指端目标
        conv, done, _ = servo.goto_lerp(
            tip_tgt, -1.0, mdp, total_steps=250,
            hook=lambda: latch_hist.append(int(_sm(env, mdp_sm).z_latch[0])))
        servo.keep_distance_buf_fresh()
        for _ in range(30):
            _, done = servo.step(tip_tgt, -1.0, mdp)
            latch_hist.append(int(_sm(env, mdp_sm).z_latch[0]))
            if done or latch_hist[-1] >= 5:
                break
        print(f"  运输后 z_latch={latch_hist[-1]} conv={conv} done={done}")
        servo.keep_distance_buf_fresh()

        # 下降插入 → c5（插值缓动；钉/座顶住即停）
        tip_now = servo.tip_mid_w(mdp)
        down = tip_now + torch.tensor([0.0, 0.0, -0.10], device=u.device)
        servo.goto_lerp(
            down, -1.0, mdp, total_steps=150,
            hook=lambda: latch_hist.append(int(_sm(env, mdp_sm).z_latch[0])))
        for _ in range(40):
            _, done = servo.step(down, -1.0, mdp)
            latch_hist.append(int(_sm(env, mdp_sm).z_latch[0]))
            if done or latch_hist[-1] >= 6:
                break
        print(f"  插入后 z_latch={latch_hist[-1]} done={done}")
        servo.keep_distance_buf_fresh()

        # 插入 settle + 尽早松爪：c6 要在 success 计数（30 帧）满之前点火
        for _ in range(25):
            _, done = servo.step(down, 1.0, mdp)
            latch_hist.append(int(_sm(env, mdp_sm).z_latch[0]))
            if done:
                break
        retreat = servo.tip_mid_w(mdp) + torch.tensor([0.0, 0.0, 0.10],
                                                      device=u.device)
        success = False
        for _ in range(120):
            _, done = servo.step(retreat, 1.0, mdp)
            latch_hist.append(int(_sm(env, mdp_sm).z_latch[0]))
            if done:
                success = bool(_sm(env, mdp_sm).last_episode_success[0])
                break
        # success 可能在插入段（lerp 下降本身准静态）就已终止 → 补读一次
        success = success or bool(
            mdp_sm._state(u).last_episode_success[0])
        servo.keep_distance_buf_fresh()

        # 单调性只统计 episode 重置前的轨迹（success 终止会自动 reset，
        # latch 归 0 是终点标志不是回跳）
        cut = len(latch_hist)
        for i, v in enumerate(latch_hist):
            if i > 0 and v == 0 and latch_hist[i - 1] > 0:
                cut = i
                break
        traj = latch_hist[:cut]
        mono = all(b >= a for a, b in zip(traj, traj[1:]))
        # latch≥6 = c5 插入锁存；c6 释放若赶在 success 前点火则到 7，
        # 两者都接受（success 不要求 c6，文档 §0）
        results[1] = mono and max(traj) >= 6 and success
        print(f"  z_latch 轨迹峰值={max(traj)} 单调={mono} "
              f"success={success} → {'通过' if results[1] else '失败'}")

    # ------------------------------------------------------------------ #
    # 测试 2：抓后松手，latch 不动、不终止不罚
    # ------------------------------------------------------------------ #
    if args.only in (0, 2):
        print("== 测试 2：抓起后松手 ==")
        servo = fresh_start()
        cup_w = _cup(env)[0].clone()
        servo.goto(cup_w + torch.tensor([0, 0, 0.12], device=u.device),
                   1.0, mdp)
        servo.goto_lerp(cup_w, 1.0, mdp, total_steps=150, tol=0.005)
        for _ in range(60):
            servo.step(cup_w, -1.0, mdp)
        lift = cup_w + torch.tensor([0, 0, 0.08], device=u.device)
        for _ in range(60):
            servo.step(lift, -1.0, mdp)
        servo.keep_distance_buf_fresh()
        latch_hold = int(_sm(env, mdp_sm).z_latch[0])
        done_before = done_or_trunc()
        for _ in range(40):                     # 低空松手（杯落回桌面）
            _, done = servo.step(lift, 1.0, mdp)
            if done:
                break
        latch_after = int(_sm(env, mdp_sm).z_latch[0])
        done_after = done_or_trunc()
        results[2] = (latch_after == latch_hold and not done_after
                      and not done_before and latch_hold >= 2)
        print(f"  latch 保持 {latch_hold}→{latch_after}，松手后无终止 "
              f"→ {'通过' if results[2] else '失败'}")

    # ------------------------------------------------------------------ #
    # 测试 3：反复夹起-放下 ×10，累计 R_first 恒定（不过不进训练）
    # ------------------------------------------------------------------ #
    if args.only in (0, 3):
        print("== 测试 3：反复夹放 10 次 ==")
        servo = fresh_start()
        cup_w0 = _cup(env)[0].clone()
        servo.goto(cup_w0 + torch.tensor([0, 0, 0.12], device=u.device),
                   1.0, mdp)
        servo.goto_lerp(cup_w0, 1.0, mdp, total_steps=150, tol=0.005)
        r_first_total = 0.0
        r_first_marks = []
        latch_peak = 0
        aborted = False
        for cycle in range(10):
            cup_w = _cup(env)[0].clone()
            for _ in range(60):                 # 闭拢夹持
                _, done = servo.step(cup_w, -1.0, mdp)
                st = _sm(env, mdp_sm)
                r_first_total += float(st._latch_delta[0]) * p.w_stage
                latch_peak = max(latch_peak, int(st.z_latch[0]))
                if done:
                    aborted = True
                    break
            if aborted:
                break
            lift = servo.tip_mid_w(mdp) + torch.tensor([0, 0, 0.06],
                                                       device=u.device)
            for _ in range(40):                 # 抬起
                _, done = servo.step(lift, -1.0, mdp)
                st = _sm(env, mdp_sm)
                r_first_total += float(st._latch_delta[0]) * p.w_stage
                latch_peak = max(latch_peak, int(st.z_latch[0]))
                if done:
                    aborted = True
                    break
            if aborted:
                break
            down = servo.tip_mid_w(mdp) + torch.tensor([0, 0, -0.06],
                                                       device=u.device)
            for _ in range(40):                 # 放回 + 松爪
                _, done = servo.step(down, 1.0, mdp)
                st = _sm(env, mdp_sm)
                r_first_total += float(st._latch_delta[0]) * p.w_stage
                if done:
                    aborted = True
                    break
            if aborted:
                break
            r_first_marks.append(r_first_total)
            servo.keep_distance_buf_fresh()
        # 判据：第 1 个循环后 R_first 不再增长（锁存不回退、不重复发钱）
        growth_after_first = (r_first_marks[-1] - r_first_marks[0]
                              if len(r_first_marks) >= 2 else float("nan"))
        results[3] = (not aborted and len(r_first_marks) >= 5
                      and abs(growth_after_first) < 1e-6
                      and latch_peak >= 3)
        print(f"  R_first 逐循环累计: {[round(x, 3) for x in r_first_marks]}"
              f"  latch 峰值={latch_peak} aborted={aborted}"
              f"  → {'通过' if results[3] else '失败'}")

    # ------------------------------------------------------------------ #
    # 测试 4：hack 探针
    # ------------------------------------------------------------------ #
    if args.only in (0, 4):
        print("== 测试 4：hack 探针 ==")
        ok4 = True
        # 4a 空手悬停杯上 50 步：c0 不得持续为真（悬停 12cm > d_aim 5cm）
        servo = fresh_start()
        cup_w = _cup(env)[0].clone()
        servo.goto(cup_w + torch.tensor([0, 0, 0.12], device=u.device),
                   1.0, mdp)
        c0_true = 0
        for _ in range(50):
            servo.step(cup_w + torch.tensor([0, 0, 0.12], device=u.device),
                       1.0, mdp)
            c0_true += int(_sm(env, mdp_sm).true[0, 0])
        print(f"  4a 空手悬停: c0 为真 {c0_true}/50 步（期望 0）")
        ok4 &= c0_true == 0
        # 4b 空夹满力 50 步：c2 不得为真（min_gap + 指-杯传感器双重拒）
        c2_true = 0
        for _ in range(50):
            servo.step(cup_w + torch.tensor([0, 0, 0.12], device=u.device),
                       -1.0, mdp)
            c2_true += int(_sm(env, mdp_sm).true[0, 2])
        print(f"  4b 空夹满力: c2 为真 {c2_true}/50 步（期望 0）")
        ok4 &= c2_true == 0
        # 4c 横扫扇飞：success 不得触发（锁存推进序 + 准静态门）
        servo = fresh_start()
        cup_w = _cup(env)[0].clone()
        sweep = cup_w + torch.tensor([0.25, 0.0, 0.02], device=u.device)
        servo.goto(cup_w + torch.tensor([-0.15, 0, 0.02], device=u.device),
                   -1.0, mdp, jam_guard=False)
        succ = False
        for _ in range(50):
            _, done = servo.step(sweep, -1.0, mdp)
            succ |= bool(_sm(env, mdp_sm).success_flag[0])
            if done:
                break
        print(f"  4c 横扫扇飞: success 触发={succ}（期望 False）")
        ok4 &= not succ

        # 4d 钉座旁静止（2026-09-21 假成功漏洞回归）：杯搬到钉 xy 4cm 内、
        # 桌面高度（z_rel=−0.12 天然满足 c5 高度条件）静止 >k_success 帧。
        # 未走链条（z_latch=0）时 success 不得触发、不得终止——成功判定
        # 必须挂 latch≥6（用户裁决：上一步成功才允许判下一步成功）。
        servo = fresh_start()
        st0 = _sm(env, mdp_sm)
        slot = st0._slot_xy[0]
        cup_obj = u.scene["cup"]
        origin = u.scene.env_origins[0]
        pose = torch.zeros(1, 7, device=u.device)
        pose[0, 0] = origin[0] + slot[0]
        pose[0, 1] = origin[1] + slot[1]
        pose[0, 2] = origin[2] + p.z_table + p.cup_half_h + 0.001
        pose[0, 3] = 1.0
        cup_obj.write_root_pose_to_sim(pose)
        cup_obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=u.device))
        succ = False
        term = False
        for _ in range(p.k_success + 20):
            _, done = servo.step(servo.tip_mid_w(mdp), 1.0, mdp)
            st = _sm(env, mdp_sm)
            succ |= bool(st.success_flag[0])
            term |= done
            if done:
                break
        latch_d = int(st.z_latch[0])
        print(f"  4d 钉座旁静止: success={succ} 终止={term} "
              f"latch={latch_d}（期望全 False/0）")
        ok4 &= (not succ) and (not term) and latch_d == 0

        results[4] = bool(ok4)
        print(f"  → {'通过' if results[4] else '失败'}")

    # ------------------------------------------------------------------ #
    # 测试 5：d_aim 边界 ±1mm 抖动，c0 不高频翻转
    # ------------------------------------------------------------------ #
    if args.only in (0, 5):
        print("== 测试 5：边界抖动 ==")
        servo = fresh_start()
        cup_w = _cup(env)[0].clone()
        base = cup_w + torch.tensor([0.0, 0.0, p.d_aim_enter],
                                    device=u.device)
        flips = 0
        prev = bool(_sm(env, mdp_sm).true[0, 0])
        for t in range(100):
            off = 0.001 if t % 2 == 0 else -0.001
            servo.step(base + torch.tensor([0, 0, off], device=u.device),
                       1.0, mdp)
            cur = bool(_sm(env, mdp_sm).true[0, 0])
            flips += int(cur != prev)
            prev = cur
        # k_enter=5 + k_exit=10：100 步全速抖也最多 ~7 次翻转
        results[5] = flips <= 10
        print(f"  c0 翻转 {flips} 次/100 步（上限 10）"
              f" → {'通过' if results[5] else '失败'}")

    # ------------------------------------------------------------------ #
    # 测试 6：随机 reset ×100，首步全谓词假、latch=0
    # ------------------------------------------------------------------ #
    if args.only in (0, 6):
        print("== 测试 6：随机 reset ==")
        bad = 0
        for _ in range(100):
            env.reset()
            env.step(torch.zeros(1, 16, device=u.device))
            st = _sm(env, mdp_sm)
            if bool(st.true[0].any()) or int(st.z_latch[0]) != 0:
                bad += 1
        results[6] = bad == 0
        print(f"  100 次 reset 后首步异常 {bad} 次"
              f" → {'通过' if results[6] else '失败'}")

    # ------------------------------------------------------------------ #
    # 测试 7：max_steps 超时，time_out 与 terminated 分离
    # ------------------------------------------------------------------ #
    if args.only in (0, 7):
        print("== 测试 7：超时分离 ==")
        env.reset()
        u.episode_length_buf.fill_(u.max_episode_length - 1)
        env.step(torch.zeros(1, 16, device=u.device))
        to = bool(u.reset_time_outs[0])
        te = bool(u.reset_terminated[0])
        results[7] = to and not te
        print(f"  timeout={to} terminated={te}（期望 True/False）"
              f" → {'通过' if results[7] else '失败'}")

    print("\n===== 汇总 =====")
    for k in sorted(results):
        print(f"  测试 {k}: {'通过 ✓' if results[k] else '失败 ✗'}")
    if args.only == 0:
        hard_ok = results.get(3, False)
        print(f"测试 3（不过不进训练）: {'通过 ✓' if hard_ok else '失败 ✗'}")
    return all(results.values())


# --------------------------------------------------------------------------- #
# --eval 模式（文档 §8）：成功率 + 部分分（max z_latch / 7）
# --------------------------------------------------------------------------- #

def run_eval(ctx, args):
    import torch

    import simulation.tasks  # noqa: F401
    from simulation.rl import facade
    from simulation.tasks.hang_cup import mdp_sm
    from simulation.tasks.hang_cup.env_cfg import load_sim_config, make_rl_cfg
    from simulation.tasks.hang_cup.agents.rsl_rl_ppo_cfg import (
        HangCupPPORunnerCfg)

    env = ctx.make_env(load_sim_config()["task_id"], cfg=make_rl_cfg(num_envs=1))
    u = env.unwrapped
    wrapped = facade.wrap_env_for_training(env)
    policy = facade.load_policy(str(args.eval), wrapped,
                                HangCupPPORunnerCfg())
    n_succ = 0
    latch_max_sum = 0
    for ep in range(args.episodes):
        obs, _ = wrapped.reset()
        latch_peak = 0
        for _ in range(u.max_episode_length):
            with torch.no_grad():
                obs, _, dones, _ = wrapped.step(policy(obs))
            st = mdp_sm.update_stage_machine(u)
            latch_peak = max(latch_peak, int(st.z_latch[0]))
            if bool(dones[0]):
                break
        n_succ += int(mdp_sm._state(u).last_episode_success[0].item())
        latch_max_sum += latch_peak
        if (ep + 1) % 10 == 0:
            print(f"  [{ep+1}/{args.episodes}] 成功率 "
                  f"{n_succ/(ep+1):.1%} 部分分 "
                  f"{latch_max_sum/(ep+1)/7:.2f}/7")
    print(f"评估 {args.episodes} episode: 成功率 {n_succ/args.episodes:.1%},"
          f" 部分分 {latch_max_sum/args.episodes/7:.2f}/7"
          f"（验收：成功率 ≥90% 且部分分 ≥5.5/7 量级——文档 §11 按 6 级，"
          f"实现平移到 7 级，对应 ≥6.4/7）")
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--only", type=int, default=0,
                    help="只跑某条测试（1-7）；0=全跑")
    ap.add_argument("--eval", type=Path, default=None,
                    help="评估模式：ckpt 路径（.pt 或含 model.pt 的目录）")
    ap.add_argument("--episodes", type=int, default=100,
                    help="评估 episode 数（文档 §8：≥100）")
    args = ap.parse_args()

    with SimContext(headless=not args.gui, enable_cameras=False) as ctx:
        ok = run_eval(ctx, args) if args.eval else run_tests(ctx, args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
