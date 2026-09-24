#!/usr/bin/env python
"""生成空心杯资产 cup_tube.usda（纯 CPU，trimesh 造视觉网格 + 手写 USDA 文本）。

背景与教训（2026-09-18，用户 GUI 两轮确诊）：
  v1 实心圆柱：钉进不了杯腔，hung 物理不可达——建模性 bug；
  v2 视觉网格直接挂 convexDecomposition：网格非水密（管+底拼接含内嵌
     面），V-HACD 烹饪出错——内壁/顶沿/底没有有效碰撞，从外侧能碰、
     从内侧抓不住，且薄壁 hull 穿模。
  v3（本版）：**视觉与碰撞解耦**——
     · 视觉：细管网格（64 段圆，渲染好看），不挂任何碰撞 API；
     · 碰撞：全部 PhysX 原生图元（Mimic/RoboCasa 做空心容器的标准
       做法）——12 块 box 围成环（内壁 r=0.025 / 外壁 r=0.035 /
       高 0.085，box 六面皆有碰撞 ⇒ 内壁、外壁、顶沿全覆盖）+
       原生圆柱底（r=0.035 / 厚 5mm ⇒ 杯底内外两面全覆盖）；
     · 碰撞体都是 /Cup 的直接子 prim 且 purpose="guide"（不参与渲染，
       但接触传感器的 "/Cup/*" glob 能匹配到——传感器 filter 依赖此结构）。
  v3+（2026-09-19 隧道效应加固）：排查证实手指/手掌碰撞凸 hull 完整
     （52/203 顶点全覆盖，侧面不缺），穿插根因是**高速隧道效应**——
     指尖速度可达 ~1.35m/s，60Hz 物理步下每步位移 ~22mm ≫ 接触边际。
     对策：杯碰撞 contactOffset 提到 8mm（speculative 边际）+
     physxRigidBody:enableCCD（杯被撞飞时不再穿过桌/架）；
     机器人侧 offset 与 120Hz 物理率见 robots.py / simulation.yaml。
  摩擦：不在资产内绑定，由 env_cfg 的 startup 事件经 root_physx_view
  设置（物理层按 shape 覆盖，全部 13 个碰撞 shape 都吃到 μ_s=1.2/μ_d=1.0）。

几何约束（2026-09-20 用户裁决：全尺寸 ×0.6——缩小杯让 panda 8cm
指距的骑跨余量从 5mm/侧放到 19mm/侧，破"蹭而不抓"（v15 确诊：
夹爪探索噪声 ±15mm/步 ≫ 5mm 余量，骑跨物理不可达））：
  外径 r=0.021、总高 0.054、原点在杯体 z 中点、质量 0.2kg（不缩）；
  腔内半径 0.015 ≫ 挂钉 r=0.005（径向余量 10mm，hung 不受影响）。
  奖励常量随之迁移：静止杯心 0.795→0.777（桌面 0.75 + 半高 0.027），
  env_cfg / mdp / 各脚本的 0.795·0.045 引用已同步。

质量说明（2026-09-19 穿模彻查裁决，烘焙进资产）：真纸杯 ~0.05kg，
但位置驱动夹爪等效 ∞ 质量，质量比 ~∞:1 时棱接触硬推（22~69N）下
PhysX 给 3~6mm 持续穿透；0.2kg + min_pos_iter=8（env_cfg）实测
5.91mm→0.44mm。重于真杯是已知失真，换接触稳定（用户裁决）。
注意：root_physx_view.set_masses 在本机 IsaacSim 5.1 后端拒绝
（"Failed to set rigid body masses in backend"），质量只能烘焙进
资产，不能运行时改。

用法（CPU 即可，不需启动仿真）：
    python src/simulation/assets/gen_cup_usda.py
产物：与本脚本同目录的 cup_tube.usda（入库提交）。
"""

import math

import numpy as np
import trimesh

from pathlib import Path

R_OUT = 0.021            # 外半径（2026-09-20 用户裁决：全尺寸 ×0.6，
                         # 原 0.035。panda 全张指距 8cm vs 杯外径 7cm 的
                         # 骑跨余量仅 5mm/侧，PPO 探索物理上命中不了；
                         # ×0.6 后杯外径 4.2cm，骑跨余量 19mm/侧）
R_IN_COLL = 0.015        # 碰撞内半径（碰撞壁厚 6mm=10mm×0.6；穿模防护
                         # 改由 contactOffset 8mm + CCD + min_pos_iter=8 承担）
HEIGHT = 0.054           # 总高（0.09×0.6）
BOTTOM = 0.003           # 底厚 3mm（5mm×0.6）
N_WALL = 12              # 碰撞环 box 数
MASS_KG = 0.2             # 不随尺寸缩：质量是穿模稳定性裁决（见 docstring），
                          # 小杯同质量密度更高、接触更稳，不是失真回归
COLOR = (0.72, 0.55, 0.38)

OUT = Path(__file__).resolve().parent / "cup_tube.usda"


def build_visual_mesh() -> trimesh.Trimesh:
    """视觉网格：管 + 底（仅渲染，不参与碰撞）。

    壁厚取与碰撞一致的 10mm（r_min=R_IN_COLL，2026-09-19 确诊）：视觉
    5mm/碰撞 10mm 的 mismatch 会让腔内捏壁的指垫恒显示 5mm 视觉穿模
    （垫停在碰撞内皮 r=2.5，视觉内皮却在 r=3.0）。对齐后腔内接触即
    视觉贴合。腔内径 5cm 仍 ≫ 挂钉 r=0.005，任务不受影响。"""
    tube = trimesh.creation.annulus(r_min=R_IN_COLL, r_max=R_OUT,
                                    height=HEIGHT - BOTTOM, sections=64)
    tube.apply_translation([0, 0, BOTTOM / 2])
    bottom = trimesh.creation.cylinder(radius=R_OUT, height=BOTTOM, sections=64)
    bottom.apply_translation([0, 0, -HEIGHT / 2 + BOTTOM / 2])
    return trimesh.util.concatenate([tube, bottom])


def _wall_box_usda(i: int) -> str:
    """第 i 块环壁 box：径向厚 10mm、弧长取中半径弦长、全高。"""
    r_mid = (R_IN_COLL + R_OUT) / 2                       # 0.030
    arc = 2 * r_mid * math.sin(math.pi / N_WALL)          # 弦长 ~0.0155
    thick = R_OUT - R_IN_COLL                             # 0.010
    wall_h = HEIGHT - BOTTOM                              # 0.085
    z = -HEIGHT / 2 + BOTTOM + wall_h / 2                 # +0.0025
    theta = 2 * math.pi * i / N_WALL
    x, y = r_mid * math.cos(theta), r_mid * math.sin(theta)
    deg = math.degrees(theta)
    return f"""
    def Cube "wall_{i:02d}" (
        apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
    )
    {{
        uniform token purpose = "guide"
        double size = 1
        float physxCollision:contactOffset = 0.008
        float physxCollision:restOffset = 0.0
        float3 xformOp:scale = ({thick:.6f}, {arc:.6f}, {wall_h:.6f})
        float xformOp:rotateZ = {deg:.3f}
        double3 xformOp:translate = ({x:.6f}, {y:.6f}, {z:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]
    }}"""


def _bottom_usda() -> str:
    """杯底：原生圆柱碰撞（PhysX 自动按凸体烹饪），盖住底内外两面。"""
    z = -HEIGHT / 2 + BOTTOM / 2
    return f"""
    def Cylinder "bottom" (
        apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
    )
    {{
        uniform token purpose = "guide"
        double radius = {R_OUT}
        double height = {BOTTOM}
        float physxCollision:contactOffset = 0.008
        float physxCollision:restOffset = 0.0
        double3 xformOp:translate = (0, 0, {z:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}"""


def _visual_mesh_usda(mesh: trimesh.Trimesh) -> str:
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    fmt_pts = ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in pts)
    counts = ", ".join("3" for _ in range(len(faces)))
    indices = ", ".join(str(i) for i in faces.reshape(-1))
    return f"""
    def Mesh "visual"
    {{
        float3[] points = [{fmt_pts}]
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]
        double3[] extent = [({lo[0]:.6f}, {lo[1]:.6f}, {lo[2]:.6f}), ({hi[0]:.6f}, {hi[1]:.6f}, {hi[2]:.6f})]
        color3f[] primvars:displayColor = [{COLOR}]
    }}"""


def write_usda(mesh: trimesh.Trimesh, path: Path) -> None:
    walls = "".join(_wall_box_usda(i) for i in range(N_WALL))
    usda = f"""#usda 1.0
(
    defaultPrim = "Cup"
    upAxis = "Z"
    metersPerUnit = 1
    doc = "空心杯 v3（gen_cup_usda.py）：视觉细网格 + 碰撞=12 box 环+原生圆柱底（全原生图元）"
)

def Xform "Cup" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI", "PhysxContactReportAPI"]
)
{{
    float physics:mass = {MASS_KG}
    bool physxRigidBody:enableCCD = 1
    float physxContactReport:threshold = 0.0
{_visual_mesh_usda(mesh)}
{walls}
{_bottom_usda()}
}}
"""
    path.write_text(usda)


def main():
    mesh = build_visual_mesh()
    lo, hi = mesh.bounds
    assert abs(hi[2] + lo[2]) < 1e-9, "原点必须在杯体 z 中点"
    assert abs(hi[0] - R_OUT) < 1e-6 and abs(hi[2] - HEIGHT / 2) < 1e-9
    write_usda(mesh, OUT)
    print(f"-> {OUT}  (视觉 {len(mesh.vertices)} 顶点 {len(mesh.faces)} 面; "
          f"碰撞 {N_WALL} box 环 + 圆柱底; 内壁 r={R_IN_COLL} 外壁 r={R_OUT})")


if __name__ == "__main__":
    main()
