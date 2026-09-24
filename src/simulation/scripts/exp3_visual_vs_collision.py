#!/usr/bin/env python
"""穿模实验 3：夹爪**视觉网格** vs **碰撞凸包**逐点对账（2026-09-19）。

动机：exp2/debug_penetration 测的是碰撞几何穿透。用户看到的是视觉穿模。
若视觉网格在接触方向上超出碰撞凸包 δ，则任何"物理上完美贴合"的抓取
都恒定显示 δ 的视觉穿模——物理怎么调都没用，必须改资产/对齐。

方法：sim 内提取 panda_leftfinger/rightfinger 的视觉 mesh（purpose
default）与碰撞 mesh（purpose guide）顶点（link 系），在指腹区域
（z > 30mm 的指尖段）逐点比较闭拢轴（link 系 y）上的超出量：
  左指接触面在 y≈0（朝 -y 闭拢），视觉 y_min < 碰撞 y_min ⇒ 超出；
  右指镜像（y_max 侧）。同时报告全轴 bbox 对比。

用法：conda run -n env_isaaclab python src/simulation/scripts/exp3_visual_vs_collision.py
"""

import sys
from pathlib import Path

_SIM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SIM_ROOT.parent))

from simulation.sim_context import SimContext  # noqa: E402

OUT = Path("outputs/debug_penetration/exp3_visual_vs_collision.log")


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    with SimContext(headless=True, enable_cameras=False) as ctx:
        import numpy as np
        import simulation.tasks  # noqa: F401
        from simulation.tasks.hang_cup.env_cfg import load_sim_config
        from simulation.tasks import default_env_cfg

        cfg = default_env_cfg(load_sim_config()["task_id"])
        cfg.scene.num_envs = 1
        for n in ("front", "left", "right"):
            delattr(cfg.scene, f"camera_{n}")
        cfg.observations.camera = None
        env = ctx.make_env(cfg=cfg, task_id=load_sim_config()["task_id"])
        env.reset()

        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = omni.usd.get_context().get_stage()

        def verts_in_link(prim, M_link_inv):
            pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),
                             dtype=np.float64)
            M = np.asarray(UsdGeom.Xformable(prim)
                           .ComputeLocalToWorldTransform(
                               Usd.TimeCode.Default()))
            return (M_link_inv @ M @ np.vstack([pts.T, np.ones(len(pts))]))[:3].T

        for link_name in ("panda_leftfinger", "panda_rightfinger"):
            lp = stage.GetPrimAtPath(
                f"/World/envs/env_0/RobotLeft/{link_name}")
            M_link_inv = np.linalg.inv(np.asarray(
                UsdGeom.Xformable(lp).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default())))
            vis_all, coll_all = [], []
            for p in Usd.PrimRange(lp, Usd.TraverseInstanceProxies()):
                if not p.IsA(UsdGeom.Mesh):
                    continue
                vv = verts_in_link(p, M_link_inv)
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    coll_all.append(vv)
                else:
                    vis_all.append(vv)
            vis = np.vstack(vis_all)
            coll = np.vstack(coll_all)
            from scipy.spatial import ConvexHull
            ch = ConvexHull(coll)
            cv = coll[ch.vertices]

            log(f"== {link_name}: 视觉 {len(vis)} 顶点, "
                f"碰撞 {len(coll)} (hull {len(cv)})")
            log(f"   视觉 bbox {vis.min(0).round(4)} ~ {vis.max(0).round(4)}")
            log(f"   碰撞 bbox {coll.min(0).round(4)} ~ {coll.max(0).round(4)}")

            # 指腹区（指尖段 z>30mm）逐视觉顶点：到碰撞凸包的**外法向超出量**
            pad = vis[vis[:, 2] > 0.030]
            eq = ch.equations                        # (F,4): n·x + b <= 0 为内
            signed = pad @ eq[:, :3].T + eq[:, 3]    # >0 = 视觉顶点在 hull 外
            outside = signed.max(axis=1)             # 每点最深外超出
            n_out = (outside > 1e-6).sum()
            log(f"   指腹区视觉顶点 {len(pad)} 个，在碰撞 hull 外 "
                f"{n_out} 个，最大超出 {outside.max() * 1000:.2f}mm, "
                f"平均(外点) {outside[outside > 1e-6].mean() * 1000 if n_out else 0:.2f}mm")
            # 闭拢轴专项：接触面方向的超出（左指 -y 侧 / 右指 +y 侧）
            if "left" in link_name:
                v_face, c_face = -pad[:, 1].max(), -coll[:, 1].min()
            else:
                v_face, c_face = pad[:, 1].max(), coll[:, 1].max()
            over = v_face - c_face
            log(f"   闭拢轴接触面：视觉 {v_face:+.4f} vs 碰撞 {c_face:+.4f}"
                f" ⇒ 视觉超出 {over * 1000:+.2f}mm"
                + ("（正值 = 抓取时恒定视觉穿模!）" if over > 0 else ""))
        env.close()

    OUT.write_text("\n".join(lines))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
