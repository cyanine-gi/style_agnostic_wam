#!/usr/bin/env python
"""穿模实验 2：楔住（rim wedging）场景的受控对照实验（2026-09-19）。

debug_penetration.py 确诊的唯一穿模机制：指尖楔在杯沿上被 IK/策略持续
硬推（22~69N），PhysX 求解器在极端质量比（位置驱动指 ≈ ∞ vs 0.05kg 杯）
+ min_pos_iter=1 下给出 3~6mm 持续性穿透。本脚本控制变量找有效解药：

  --iters N   改 sim physx min_position_iteration_count（默认 1）
  --mass  M   把杯质量改成 M kg（默认不改 = 0.05）
  --stab      开 enable_stabilization

每组跑同一确定性场景：悬停 → 对准杯心下降（目标在杯心下方 2cm，保证
楔住）→ 楔住保持 30 步。逐步记录：穿透深度(mm)、指-杯力、cos、
_clamp 判定、grip_force 奖励——后三者验证"楔住是否会被奖励链白拿"。

用法（每组一个进程）：
    conda run -n env_isaaclab python src/simulation/scripts/exp2_wedging.py \
        --iters 8 --mass 0.2 --tag iter8_mass02
"""

import argparse
import math
import sys
from pathlib import Path

_SIM_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _SIM_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))

from simulation.sim_context import SimContext  # noqa: E402

R_OUT, R_IN, HEIGHT, BOTTOM, N_WALL = 0.035, 0.025, 0.09, 0.005, 12
OUTDIR = Path("outputs/debug_penetration")


def wall_boxes_local():
    r_mid = (R_IN + R_OUT) / 2
    arc = 2 * r_mid * math.sin(math.pi / N_WALL)
    thick = R_OUT - R_IN
    wall_h = HEIGHT - BOTTOM
    z = -HEIGHT / 2 + BOTTOM + wall_h / 2
    out = []
    for i in range(N_WALL):
        th = 2 * math.pi * i / N_WALL
        out.append(((r_mid * math.cos(th), r_mid * math.sin(th), z),
                    th, (thick / 2, arc / 2, wall_h / 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--mass", type=float, default=None)
    ap.add_argument("--stab", action="store_true")
    ap.add_argument("--grip-stiffness", type=float, default=None,
                    help="改夹爪驱动刚度（HIGH_PD 默认 2000）")
    ap.add_argument("--tag", type=str, default="baseline")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--pin-cup", action="store_true")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    with SimContext(headless=True, enable_cameras=args.render) as ctx:
        import numpy as np
        import torch
        import simulation.tasks  # noqa: F401
        from isaaclab.controllers import DifferentialIKController
        from isaaclab.controllers.differential_ik_cfg import (
            DifferentialIKControllerCfg)
        from isaaclab.utils.math import (matrix_from_quat, quat_apply,
                                         quat_inv,
                                         subtract_frame_transforms)
        from simulation.tasks.hang_cup import mdp
        from simulation.tasks.hang_cup.env_cfg import load_sim_config
        from simulation.tasks import default_env_cfg

        task_id = load_sim_config()["task_id"]
        cfg = default_env_cfg(task_id)
        cfg.scene.num_envs = 1
        # 无相机训练款（同 make_rl_cfg）
        for n in ("front", "left", "right"):
            delattr(cfg.scene, f"camera_{n}")
        cfg.observations.camera = None
        if args.render:
            import isaaclab.sim as sim_utils
            from isaaclab.sensors import CameraCfg
            setattr(cfg.scene, "camera_debug", CameraCfg(
                prim_path="{ENV_REGEX_NS}/Camera_debug",
                update_period=0.0, update_latest_camera_pose=True,
                width=320, height=240, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=22.0,
                                                 clipping_range=(0.02, 10.0))))
        if args.iters is not None:
            cfg.sim.physx.min_position_iteration_count = args.iters
            cfg.sim.physx.min_velocity_iteration_count = max(1, args.iters // 2)
        if args.stab:
            cfg.sim.physx.enable_stabilization = True
        if args.mass is not None:
            # 烘焙质量变体资产（set_masses 后端拒绝——2026-09-19 两轮实测），
            # 直接 sed USDA 的 physics:mass 到 /tmp 变体再指过去
            src_usd = Path(cfg.scene.cup.spawn.usd_path)
            variant = Path(f"/tmp/cup_tube_mass{args.mass}.usda")
            txt = src_usd.read_text()
            import re as _re
            txt2 = _re.sub(r"float physics:mass = [0-9.]+",
                           f"float physics:mass = {args.mass}", txt)
            assert txt2 != txt, "mass 替换失败"
            variant.write_text(txt2)
            cfg.scene.cup.spawn.usd_path = str(variant)
        if args.grip_stiffness is not None:
            for robot_name in ("robot_left", "robot_right"):
                arms = getattr(cfg.scene, robot_name)
                arms.actuators["panda_hand"].stiffness = args.grip_stiffness
                arms.actuators["panda_hand"].damping = max(
                    20.0, args.grip_stiffness * 0.05)

        env = ctx.make_env(task_id, cfg=cfg)
        u = env.unwrapped
        dev = u.device
        side = "left"
        g_dim, arm_slice = 14, slice(0, 7)
        robot = u.scene["robot_left"]
        cup = u.scene["cup"]
        arm_ids, _ = robot.find_joints(
            [f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)
        hand_id = robot.find_bodies("panda_hand")[0][0]
        lf_id = robot.find_bodies("panda_leftfinger")[0][0]
        rf_id = robot.find_bodies("panda_rightfinger")[0][0]

        log(f"config: tag={args.tag} iters={args.iters} mass={args.mass} "
            f"stab={args.stab} grip_k={args.grip_stiffness}")
        log(f"  实际 min_pos_iter={u.sim.cfg.physx.min_position_iteration_count}"
            f" min_vel_iter={u.sim.cfg.physx.min_velocity_iteration_count}"
            f" stab={u.sim.cfg.physx.enable_stabilization}")

        env.reset()
        zeros = torch.zeros(1, 16, device=dev)
        for _ in range(30):
            env.step(zeros)
        if args.pin_cup:
            fixed = torch.tensor([[0.05, -0.02, 0.75 + 0.045, 1.0, 0, 0, 0]],
                                 device=dev)
            cup.write_root_pose_to_sim(fixed[:, :7])
            cup.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
            for _ in range(15):
                env.step(zeros)

        if args.mass is not None:
            log(f"  杯质量实测: {cup.root_physx_view.get_masses().tolist()}")

        # ---- 指碰撞 hull 顶点（link 系，同 debug_penetration 的方法）----
        import omni.usd
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        from scipy.spatial import ConvexHull

        finger_hull = {}
        for link_name in ("panda_leftfinger", "panda_rightfinger"):
            coll = stage.GetPrimAtPath(
                f"/World/envs/env_0/RobotLeft/{link_name}/collisions/collisions")
            lp = stage.GetPrimAtPath(
                f"/World/envs/env_0/RobotLeft/{link_name}")
            pts = np.asarray(UsdGeom.Mesh(coll).GetPointsAttr().Get(),
                             dtype=np.float64)
            M_mesh = np.asarray(UsdGeom.Xformable(coll)
                                .ComputeLocalToWorldTransform(
                                    Usd.TimeCode.Default()))
            M_link = np.asarray(UsdGeom.Xformable(lp)
                                .ComputeLocalToWorldTransform(
                                    Usd.TimeCode.Default()))
            vv = (np.linalg.inv(M_link) @ M_mesh @
                  np.vstack([pts.T, np.ones(len(pts))]))[:3].T
            hull = ConvexHull(vv)
            finger_hull[link_name] = (
                torch.tensor(vv[hull.vertices], dtype=torch.float32,
                             device=dev))

        wb = wall_boxes_local()
        box_c = torch.tensor([b[0] for b in wb], dtype=torch.float32,
                             device=dev)
        box_th = [b[1] for b in wb]
        box_h = torch.tensor([b[2] for b in wb], dtype=torch.float32,
                             device=dev)

        def pen_mm():
            """指 hull 顶点 vs 12 壁 box 的最大穿透（mm）。"""
            from isaaclab.utils.math import quat_apply as qa
            cp, cq = cup.data.root_pos_w[0], cup.data.root_quat_w[0]
            cq_inv = torch.stack([cq[0], -cq[1], -cq[2], -cq[3]])
            worst = 0.0
            for link_name, f_id in (("panda_leftfinger", lf_id),
                                    ("panda_rightfinger", rf_id)):
                vv = finger_hull[link_name]
                bp = robot.data.body_pos_w[0, f_id]
                bq = robot.data.body_quat_w[0, f_id]
                vw = qa(bq[None], vv) + bp[None]
                vc = qa(cq_inv[None], vw - cp[None])
                for i in range(12):
                    loc = vc - box_c[i]
                    c, s = math.cos(-box_th[i]), math.sin(-box_th[i])
                    x = c * loc[:, 0] - s * loc[:, 1]
                    y = s * loc[:, 0] + c * loc[:, 1]
                    ax = torch.stack([x, y, loc[:, 2]], -1).abs()
                    inside = (ax < box_h[i]).all(dim=-1)
                    if inside.any():
                        d = (box_h[i] - ax[inside]).min(-1).values.max()
                        worst = max(worst, d.item() * 1000)
            return worst

        # ---- IK 伺服（同前）----
        ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose",
                                        use_relative_mode=False,
                                        ik_method="dls"),
            num_envs=1, device=dev)
        ik.reset()
        origins = u.scene.env_origins

        def ee_pose_b():
            return subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w,
                robot.data.body_pos_w[:, hand_id],
                robot.data.body_quat_w[:, hand_id])

        def jacobian_b():
            J = robot.root_physx_view.get_jacobians()[
                :, hand_id - 1, :, arm_ids].clone()
            R = matrix_from_quat(quat_inv(robot.data.root_quat_w))
            J[:, :3, :] = torch.bmm(R, J[:, :3, :])
            J[:, 3:, :] = torch.bmm(R, J[:, 3:, :])
            return J

        def tip_mid_w():
            return mdp._finger_positions(u, side)[0].mean(dim=0) + origins[0]

        def w2b(p_w):
            return torch.bmm(
                matrix_from_quat(quat_inv(robot.data.root_quat_w)),
                (p_w - robot.data.root_pos_w).unsqueeze(-1)).squeeze(-1)

        hold_quat_b = ee_pose_b()[1].clone()
        tip_off_b = (ee_pose_b()[0] - w2b(tip_mid_w())).clone()

        def ik_step(tgt_w, grip_raw):
            ep, eq = ee_pose_b()
            q_cur = robot.data.joint_pos[:, arm_ids]
            cmd_pos = w2b(tgt_w) + tip_off_b
            ik.set_command(torch.cat([cmd_pos, hold_quat_b], dim=-1))
            q_des = ik.compute(ep, eq, jacobian_b(), q_cur)
            a = zeros.clone()
            a[0, arm_slice] = ((q_des - q_cur) / 0.1).clamp(-1.0, 1.0)[0]
            a[0, g_dim] = grip_raw
            env.step(a)

        # ---- 确定性楔住场景 ----
        cup_w = cup.data.root_pos_w[0].clone()
        hover = cup_w + torch.tensor([0.0, 0.0, 0.12], device=dev)
        wedge = cup_w - torch.tensor([0.0, 0.0, 0.02], device=dev)  # 杯心下 2cm

        for _ in range(200):
            err = torch.linalg.norm(tip_mid_w() - hover).item()
            if err < 0.008:
                break
            ik_step(hover, 1.0)
        log(f"P1 hover err={err * 100:.1f}cm")

        cam = u.scene.sensors["camera_debug"] if args.render else None

        def snap(t, pen):
            if cam is None:
                return
            import cv2
            cp = cup.data.root_pos_w[0]
            eye = cp + torch.tensor([0.28, -0.28, 0.14], device=dev)
            cam.set_world_poses_from_view(
                eyes=eye[None], targets=cp[None],
                env_ids=torch.tensor([0], device=dev))
            rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
            im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(im, f"{args.tag} t={t} pen={pen:.1f}mm", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(OUTDIR / f"exp2_{args.tag}_t{t:03d}.png"), im)

        log("step  pen_mm  F1_N  F2_N   cos  clamp grip_force  grip_cm")
        rows = []
        for t in range(45):
            ik_step(wedge, 1.0)                # 张爪硬压（RL 楔住形态）
            pen = pen_mm()
            V = mdp._finger_cup_force_vec(u, side)[0]
            f = torch.linalg.norm(V, dim=-1)
            vn = V / f.clamp_min(1e-6)[:, None]
            cos = (vn[0] * vn[1]).sum().item()
            clamp, _ = mdp._clamp(u, side)
            # 2026-09-21 路线 A：rew_grip_hold 已删，本列改打 _clamp 指示
            # （同一三条件判据，语义等价）
            gf = float(clamp[0].item())
            grip = u.action_manager.get_term("arm")._current16()[0, g_dim]
            rows.append((t, pen, f[0].item(), f[1].item(), cos,
                         int(clamp[0]), gf, grip.item()))
            snap(t, pen)
            if t % 3 == 0:
                log(f"{t:4d}  {pen:6.2f} {f[0]:5.1f} {f[1]:5.1f} {cos:+5.2f}"
                    f"   {int(clamp[0])}    {gf:5.3f}   {grip.item() * 100:4.1f}")

        pens = [r[1] for r in rows]
        fs = [max(r[2], r[3]) for r in rows]
        clamps = sum(r[5] for r in rows)
        log(f"== {args.tag}: pen max={max(pens):.2f}mm "
            f"mean(后30步)={sum(pens[15:]) / 30:.2f}mm "
            f"F max={max(fs):.1f}N clamp 触发 {clamps}/45 步")

    (OUTDIR / f"exp2_{args.tag}.log").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
