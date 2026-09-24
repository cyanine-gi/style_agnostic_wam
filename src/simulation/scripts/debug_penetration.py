#!/usr/bin/env python
"""杯-夹爪穿模彻查（2026-09-19，不信任任何旧结论，从零实测）。

四阶段：
  S0 碰撞体检视：在加载后的 stage 上逐 prim 打印杯（12 box 环+底）与
     夹爪碰撞网格的真实属性（CollisionAPI/offset/purpose/材质），并打印
     PhysX 求解器参数——凡是 USD/配置里"以为设置上了"的都以 stage 实测为准。
  S1 静态对账：从 USD 提取夹爪碰撞凸包顶点，与杯壁 box 环几何对账
     （顶点间距 vs 壁厚 10mm——顶点稀于壁厚则薄壁穿透测不出来也挡不出来）。
  S2 IK 正常操作流（悬停→下降→闭爪→抬升）：逐控制步定量测穿模
     （指凸包顶点∈壁box / 壁box角点∈指凸包 双向测试），特写相机逐帧存图。
  S3 RL 式暴力随机 delta 动作：统计穿模事件（深度>1mm）与最深穿透。

判定语义（写死，不解释）：
  pen_mm > 0  = 几何穿透发生；>1mm = 显著穿模；
  pen>1mm 且指-杯力≈0 = 接触漏检（物理上真穿过去了）。

用法：
    conda run -n env_isaaclab python src/simulation/scripts/debug_penetration.py
"""

import argparse
import math
import sys
from pathlib import Path

_SIM_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _SIM_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))

from simulation.sim_context import SimContext  # noqa: E402

# 杯碰撞几何常量（与 gen_cup_usda.py 一致——这里独立重算，不 import 它，
# 防止"两边同错"的对账失效）
R_OUT, R_IN, HEIGHT, BOTTOM, N_WALL = 0.035, 0.025, 0.09, 0.005, 12
OUTDIR = Path("outputs/debug_penetration")


def wall_boxes_local():
    """杯系 12 壁 box：(center(3), theta_rad, half(3))。独立重算。"""
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
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--violent-steps", type=int, default=240)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--pin-cup", action="store_true",
                    help="S2 前把杯钉到固定位姿（A/B 对比确定性）")
    ap.add_argument("--mass", type=float, default=None)
    ap.add_argument("--grip-stiffness", type=float, default=None)
    ap.add_argument("--render", action="store_true",
                    help="加特写相机逐帧存图（需要 GPU 显存余量 ~2GB；"
                         "默认关 = 纯物理定量诊断，可在训练占卡时跑）")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(str(s))

    with SimContext(headless=True, enable_cameras=args.render) as ctx:
        import numpy as np
        import torch

        import isaaclab.sim as sim_utils
        import simulation.tasks  # noqa: F401
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils.math import (matrix_from_quat, quat_inv,
                                         subtract_frame_transforms)
        from isaaclab.controllers import DifferentialIKController
        from isaaclab.controllers.differential_ik_cfg import (
            DifferentialIKControllerCfg)
        from simulation.tasks.hang_cup import mdp
        from simulation.tasks.hang_cup.env_cfg import (
            camera_names, load_sim_config)
        from simulation.tasks import default_env_cfg

        task_id = load_sim_config()["task_id"]
        cfg = default_env_cfg(task_id)
        cfg.scene.num_envs = 1
        # YAML 相机全删掉（不用）；camera 观测组必须同时置 None，
        # 否则 ObservationManager 找 camera_front 直接 KeyError
        for n in camera_names():
            delattr(cfg.scene, f"camera_{n}")
        cfg.observations.camera = None
        if args.iters is not None:
            cfg.sim.physx.min_position_iteration_count = args.iters
            cfg.sim.physx.min_velocity_iteration_count = max(1, args.iters // 2)
        if args.mass is not None:
            import re as _re
            variant = Path(f"/tmp/cup_tube_mass{args.mass}.usda")
            txt = Path(cfg.scene.cup.spawn.usd_path).read_text()
            txt2 = _re.sub(r"float physics:mass = [0-9.]+",
                           f"float physics:mass = {args.mass}", txt)
            assert txt2 != txt
            variant.write_text(txt2)
            cfg.scene.cup.spawn.usd_path = str(variant)
        if args.grip_stiffness is not None:
            for rn in ("robot_left", "robot_right"):
                arms = getattr(cfg.scene, rn)
                arms.actuators["panda_hand"].stiffness = args.grip_stiffness
                arms.actuators["panda_hand"].damping = max(
                    20.0, args.grip_stiffness * 0.05)
        if args.render:
            setattr(cfg.scene, "camera_debug", CameraCfg(
                prim_path="{ENV_REGEX_NS}/Camera_debug",
                update_period=0.0,
                update_latest_camera_pose=True,
                width=320, height=240,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=22.0,
                                                 clipping_range=(0.02, 10.0)),
            ))
        env = ctx.make_env(task_id, cfg=cfg)
        u = env.unwrapped
        dev = u.device
        side = args.side
        g_dim = 14 if side == "left" else 15
        arm_slice = slice(0, 7) if side == "left" else slice(7, 14)

        robot = u.scene[f"robot_{side}"]
        arm_ids, _ = robot.find_joints(
            [f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)
        hand_id = robot.find_bodies("panda_hand")[0][0]
        lf_id = robot.find_bodies("panda_leftfinger")[0][0]
        rf_id = robot.find_bodies("panda_rightfinger")[0][0]

        cam = u.scene.sensors["camera_debug"] if args.render else None

        # ================================================================ #
        # S0: stage 碰撞体检视（实测，不猜）
        # ================================================================ #
        log("=" * 70)
        log("S0: stage 碰撞体检视")
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
        stage = omni.usd.get_context().get_stage()
        # 机器人是 instanceable USD——默认 Traverse 不进实例代理，
        # 必须显式带 TraverseInstanceProxies 谓词（2026-09-19 第一轮
        # 教训：不带时杯（非实例）能遍历到、夹爪碰撞体整棵不可见，
        # 穿模测量因无指顶点而静默恒 0）
        _PROXY = Usd.TraverseInstanceProxies()

        def prim_info(p):
            col = p.HasAPI(UsdPhysics.CollisionAPI)
            co = ro = None
            if p.HasAPI(PhysxSchema.PhysxCollisionAPI):
                api = PhysxSchema.PhysxCollisionAPI(p)
                co = api.GetContactOffsetAttr().Get()
                ro = api.GetRestOffsetAttr().Get()
            purpose = UsdGeom.Imageable(p).GetPurposeAttr().Get() \
                if p.IsA(UsdGeom.Imageable) else "-"
            return f"    {p.GetPath()} [{p.GetTypeName()}] col={col} " \
                   f"contactOff={co} restOff={ro} purpose={purpose}"

        log("  -- 杯 /World/envs/env_0/Cup 子树 --")
        cup_prim = stage.GetPrimAtPath("/World/envs/env_0/Cup")
        assert cup_prim.IsValid(), "Cup prim 不存在!"
        for p in stage.Traverse(_PROXY):
            sp = str(p.GetPath())
            if sp.startswith("/World/envs/env_0/Cup"):
                log(prim_info(p))

        log("  -- 夹爪碰撞 prim（RobotLeft 全树含 finger 的碰撞体）--")
        finger_coll_prims = {}
        for p in stage.Traverse(_PROXY):
            sp = str(p.GetPath())
            if not sp.startswith("/World/envs/env_0/RobotLeft/"):
                continue
            if "finger" not in sp:
                continue
            if p.HasAPI(UsdPhysics.CollisionAPI):
                approx = "-"
                if p.HasAPI(UsdPhysics.MeshCollisionAPI):
                    approx = UsdPhysics.MeshCollisionAPI(
                        p).GetApproximationAttr().Get()
                log(prim_info(p) + f" approx={approx}")
                if p.IsA(UsdGeom.Mesh):
                    # 归到所属 link（路径里含 left/rightfinger）
                    link = ("panda_leftfinger" if "leftfinger" in sp
                            else "panda_rightfinger")
                    finger_coll_prims.setdefault(link, []).append(p)

        log("  -- 杯材质（root_physx_view 实测，(13 shape, 3 属性)）--")
        cup = u.scene["cup"]
        mats = cup.root_physx_view.get_material_properties()
        m = mats[0] if isinstance(mats, (tuple, list)) else mats
        for j, name in enumerate(("static_friction", "dynamic_friction",
                                  "restitution")):
            col = m[:, j] if m.ndim == 2 else m
            log(f"    {name}: unique={torch.unique(col).tolist()}")

        log("  -- 求解器/步长 --")
        physx_cfg = u.sim.cfg.physx
        for k in ("solver_type", "min_position_iteration_count",
                  "max_position_iteration_count",
                  "min_velocity_iteration_count",
                  "max_velocity_iteration_count",
                  "bounce_threshold_velocity", "enable_ccd",
                  "enable_stabilization"):
            log(f"    {k} = {getattr(physx_cfg, k, '<无此字段>')}")
        log(f"    physics_dt = {u.sim.get_physics_dt()} "
            f"(= {1.0 / u.sim.get_physics_dt():.1f} Hz), decimation={u.cfg.decimation}")
        try:
            log(f"    articulation pos iters = "
                f"{robot.root_physx_view.get_solver_position_iteration_count()}")
        except Exception as e:
            log(f"    articulation iters 读取失败: {e}")

        # ================================================================ #
        # S1: 静态几何对账
        # ================================================================ #
        log("=" * 70)
        log("S1: 夹爪碰撞凸包 vs 杯壁几何对账")
        env.reset()
        zeros = torch.zeros(1, 16, device=dev)
        for _ in range(30):
            env.step(zeros)
        if args.pin_cup:
            from isaaclab.utils.math import quat_apply  # noqa
            fixed = torch.tensor([[0.05, -0.02, 0.75 + 0.045, 1.0, 0, 0, 0]],
                                 device=dev)
            cup = u.scene["cup"]
            cup.write_root_pose_to_sim(fixed[:, :7])
            cup.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
            for _ in range(15):
                env.step(zeros)
            log(f"  杯钉到固定位姿: {cup.data.root_pos_w[0].tolist()}")

        # 指碰撞 mesh 顶点（link 系）
        def mesh_verts_in_link(prim, link_prim):
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            M_mesh = np.asarray(UsdGeom.Xformable(prim)
                                .ComputeLocalToWorldTransform(
                                    Usd.TimeCode.Default()))
            M_link = np.asarray(UsdGeom.Xformable(link_prim)
                                .ComputeLocalToWorldTransform(
                                    Usd.TimeCode.Default()))
            return (np.linalg.inv(M_link) @ M_mesh @
                    np.vstack([pts.T, np.ones(len(pts))]))[:3].T

        finger_verts_link = {}
        link_prims = {}
        for link_name in ("panda_leftfinger", "panda_rightfinger"):
            lp = stage.GetPrimAtPath(
                f"/World/envs/env_0/RobotLeft/{link_name}")
            assert lp.IsValid(), f"{link_name} prim 不存在"
            link_prims[link_name] = lp
            prims = finger_coll_prims.get(link_name)
            if not prims:
                # 碰撞可能就在 link prim 自身
                if lp.HasAPI(UsdPhysics.CollisionAPI) and \
                        lp.IsA(UsdGeom.Mesh):
                    prims = [lp]
                else:
                    log(f"  !! {link_name} 下没找到碰撞 Mesh prim")
                    continue
            vv = np.vstack([mesh_verts_in_link(p, lp) for p in prims])
            finger_verts_link[link_name] = vv
            lo, hi = vv.min(0), vv.max(0)
            # 顶点最大间隙（最近邻距的上界抽样）
            from scipy.spatial import ConvexHull, cKDTree
            try:
                hull = ConvexHull(vv)
                hv = vv[hull.vertices]
                nn = cKDTree(hv)
                d, _ = nn.query(hv, k=2)
                log(f"  {link_name}: 碰撞顶点 {len(vv)} 个（hull {len(hv)}）"
                    f" bbox=[{lo.round(4)}]~[{hi.round(4)}]"
                    f" 顶点最近邻距 max={d.max() * 1000:.1f}mm")
                finger_verts_link[link_name + "_hull_eq"] = hull.equations
                finger_verts_link[link_name + "_hull_v"] = hv
            except Exception as e:
                log(f"  {link_name}: hull 构建失败 {e}")

        # 壁 box 数据（torch）
        wb = wall_boxes_local()
        box_c = torch.tensor([b[0] for b in wb], dtype=torch.float32,
                             device=dev)                     # (12,3)
        box_th = torch.tensor([b[1] for b in wb], dtype=torch.float32,
                              device=dev)
        box_h = torch.tensor([b[2] for b in wb], dtype=torch.float32,
                             device=dev)                     # (12,3)
        # box 8 角点（杯系）
        corners_l = []
        for i in range(12):
            hx, hy, hz = box_h[i].tolist()
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corners_l.append([sx * hx, sy * hy, sz * hz])
        corners_l = torch.tensor(corners_l, dtype=torch.float32,
                                 device=dev).view(12, 8, 3)

        def quat_rot_inv(q, v):
            """v (...,3) 用四元数 q (wxyz) 的逆旋转。"""
            from isaaclab.utils.math import quat_apply
            return quat_apply(torch.stack([q[..., 0], -q[..., 1],
                                           -q[..., 2], -q[..., 3]], dim=-1), v)

        def measure_penetration():
            """当前状态下两指 vs 杯壁/底的最大穿透深度（mm）与事件明细。

            双向：①指顶点∈box；②box角点∈指凸包。返回 (max_mm, detail)。
            """
            cup_pos = cup.data.root_pos_w[0]
            cup_q = cup.data.root_quat_w[0]
            max_pen = 0.0
            detail = ""
            for link_name, f_id in (("panda_leftfinger", lf_id),
                                    ("panda_rightfinger", rf_id)):
                key = link_name
                if key not in finger_verts_link:
                    continue
                vv = torch.tensor(finger_verts_link[key],
                                  dtype=torch.float32, device=dev)
                bp = robot.data.body_pos_w[0, f_id]
                bq = robot.data.body_quat_w[0, f_id]
                # 世界 → 杯系
                from isaaclab.utils.math import quat_apply
                vw = quat_apply(bq[None], vv) + bp[None]          # (V,3)
                vc = quat_rot_inv(cup_q, vw - cup_pos[None])      # 杯系
                # 顶点 ∈ 各 box
                for i in range(12):
                    loc = vc - box_c[i]
                    c, s = math.cos(-box_th[i].item()), math.sin(-box_th[i].item())
                    x = c * loc[:, 0] - s * loc[:, 1]
                    y = s * loc[:, 0] + c * loc[:, 1]
                    z = loc[:, 2]
                    ax = torch.stack([x, y, z], dim=-1).abs()
                    inside = (ax < box_h[i]).all(dim=-1)
                    if inside.any():
                        depth = (box_h[i] - ax[inside]).min(dim=-1).values.max()
                        d_mm = depth.item() * 1000
                        if d_mm > max_pen:
                            max_pen = d_mm
                            detail = f"{link_name} 顶点∈wall_{i:02d}"
                # 底圆柱：r<R_OUT 且 z∈[-0.045,-0.040]
                r = torch.linalg.norm(vc[:, :2], dim=-1)
                zin = (vc[:, 2] > -HEIGHT / 2) & (vc[:, 2] < -HEIGHT / 2 + BOTTOM)
                deep = (r < R_OUT) & zin
                if deep.any():
                    d_mm = (R_OUT - r[deep]).max().item() * 1000
                    if d_mm > max_pen:
                        max_pen = d_mm
                        detail = f"{link_name} 顶点∈杯底"
                # box 角点 ∈ 指凸包（反向：薄壁穿透而顶点稀疏时兜底）
                eq_key = key + "_hull_eq"
                if eq_key in finger_verts_link:
                    eq = torch.tensor(finger_verts_link[eq_key],
                                      dtype=torch.float32, device=dev)
                    # 角点（杯系）→ 世界 → link 系
                    cw_all = []
                    for i in range(12):
                        th = box_th[i].item()
                        c, s = math.cos(th), math.sin(th)
                        cl = corners_l[i]                     # (8,3)
                        x = c * cl[:, 0] - s * cl[:, 1]
                        y = s * cl[:, 0] + c * cl[:, 1]
                        cc = torch.stack([x, y, cl[:, 2]], -1) + box_c[i]
                        cw_all.append(cc)
                    cc = torch.cat(cw_all)                    # (96,3) 杯系
                    from isaaclab.utils.math import quat_apply
                    cw = quat_apply(cup_q[None], cc) + cup_pos[None]
                    cl_ = quat_rot_inv(bq, cw - bp[None])     # link 系
                    signed = cl_ @ eq[:, :3].T + eq[:, 3]     # (96,F)
                    inside = (signed < 0).all(dim=-1)
                    if inside.any():
                        depth = (-signed[inside]).min(dim=-1).values.max()
                        d_mm = depth.item() * 1000
                        if d_mm > max_pen:
                            max_pen = d_mm
                            detail = f"壁角点∈{link_name} hull"
            return max_pen, detail

        # 特写相机跟踪杯
        def track_cam():
            if cam is None:
                return
            cp = cup.data.root_pos_w[0]
            eye = cp + torch.tensor([0.30, -0.30, 0.16], device=dev)
            cam.set_world_poses_from_view(
                eyes=eye[None], targets=cp[None],
                env_ids=torch.tensor([0], device=dev))

        frame_i = [0]

        def snap(tag, pen_mm, extra=""):
            if cam is None:
                return
            import cv2
            track_cam()
            rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
            im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(im, f"{tag} pen={pen_mm:.1f}mm {extra}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            p = OUTDIR / f"{frame_i[0]:04d}_{tag}.png"
            cv2.imwrite(str(p), im)
            frame_i[0] += 1

        # IK 伺服（同 check_clamp 的 DLS 链）
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
            return torch.linalg.norm(tip_mid_w() - tgt_w).item()

        hold_quat_b = ee_pose_b()[1].clone()
        tip_off_b = (ee_pose_b()[0] - w2b(tip_mid_w())).clone()
        cup_w = cup.data.root_pos_w[0].clone()
        hover = cup_w + torch.tensor([0.0, 0.0, 0.12], device=dev)
        grasp = cup_w.clone()
        lift = grasp + torch.tensor([0.0, 0.0, 0.15], device=dev)

        # ================================================================ #
        # S2: IK 正常操作流，逐步测穿模
        # ================================================================ #
        log("=" * 70)
        log("S2: IK 操作流（每步穿模测量，>0.5mm 记事件）")
        events = []

        def run_phase(tag, tgt, grip, n_steps, snap_every=2):
            worst = (0.0, "")
            for t in range(n_steps):
                err = ik_step(tgt, grip)
                pen, det = measure_penetration()
                f_f = mdp._finger_cup_force(u, side)[0].item()
                if pen > worst[0]:
                    worst = (pen, det)
                if pen > 0.5:
                    events.append((tag, t, pen, det, f_f))
                    snap(f"{tag}_t{t:03d}_PEN", pen, det)
                elif t % snap_every == 0:
                    snap(f"{tag}_t{t:03d}", pen)
                if err < 0.005 and tag != "A_close":
                    break
            log(f"  [{tag}] 最深穿透 {worst[0]:.2f}mm {worst[1]}")

        run_phase("P1_hover", hover, 1.0, 200, snap_every=40)
        run_phase("P2_descend", grasp, 1.0, 150, snap_every=3)
        run_phase("A_close", grasp, -1.0, 45, snap_every=3)
        run_phase("B_lift", lift, -1.0, 80, snap_every=8)

        # ================================================================ #
        # S3: RL 式暴力随机 delta 动作
        # ================================================================ #
        log("=" * 70)
        log("S3: 暴力随机动作（|a|≤1，穿模事件统计）")
        env.reset()
        for _ in range(30):
            env.step(zeros)
        rng = torch.Generator(device=dev).manual_seed(7)
        n_evt = 0
        worst = (0.0, "")
        for t in range(args.violent_steps):
            a = 0.9 * torch.randn(zeros.shape, generator=rng, device=dev)
            env.step(a)
            pen, det = measure_penetration()
            f_f = mdp._finger_cup_force(u, side)[0].item()
            if pen > worst[0]:
                worst = (pen, det)
                snap(f"S3_t{t:03d}_WORST", pen, det)
            if pen > 1.0:
                n_evt += 1
                events.append(("S3", t, pen, det, f_f))
                if n_evt <= 10:
                    snap(f"S3_t{t:03d}_PEN", pen,
                         det + f" F={f_f:.1f}N")
        log(f"  [S3] 穿模事件(>1mm) {n_evt}/{args.violent_steps} 步, "
            f"最深 {worst[0]:.2f}mm {worst[1]}")

        # ================================================================ #
        # 汇总
        # ================================================================ #
        log("=" * 70)
        log("汇总: 全部穿模事件(>0.5mm):")
        for tag, t, pen, det, f_f in events[:60]:
            flag = "  <-- 接触漏检!" if pen > 1.0 and f_f < 0.5 else ""
            log(f"  {tag:10s} t={t:3d} pen={pen:6.2f}mm {det} "
                f"指-杯力={f_f:.1f}N{flag}")
        if len(events) > 60:
            log(f"  ... 共 {len(events)} 条，仅列前 60")

    (OUTDIR / "log.txt").write_text("\n".join(log_lines))
    print(f"-> {OUTDIR}/log.txt 与帧图 {frame_i[0]} 张")


if __name__ == "__main__":
    main()
