"""hang_cup 任务的环境装配：场景 + MDP 配置 + EnvCfg。

任务（复刻 RoboMIND hang_cup_on_cup_holder 的抽象版）：桌面左侧随机位置
放一个杯子，左臂抓起挂到杯架钉上。道具全部是 primitive 几何体
（2026-09-13 裁决：不追求 photo-realistic；视觉信号在环但写实度随意）。

相机：多机位（cameras 下每个键 = 场景成员 camera_<键> = HDF5 相机名），
位姿从 configs/simulation.yaml 读取——front 近似 real 0520 批斜视机位，
细调用 scripts/tune_camera.py。2026-09-17 sawwam 裁决：2–3 路同录、
每集位姿/焦距随机化（recording.py 消费 YAML 的 randomize 块）、
K/T 逐集落盘。加机位 = YAML cameras 下加条目即可，无需改本文件。

机器人：由 robots.py 注册表构建（当前 franka_dual = 两台 panda 实例，
桌面左右侧对称安装面向中线）。
"""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import numpy as np
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp as base_mdp

from simulation.robots import get_spec
from . import mdp, mdp_sm

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "simulation.yaml"
TABLE_Z = 0.75          # 桌面高度（场景几何的唯一事实源）
RACK_X, RACK_Y = 0.25, -0.20   # 杯架中心（近侧右侧，奖励/终止经由 peg 引用）


def load_sim_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def camera_names() -> list[str]:
    """YAML cameras 的键序（= 场景成员 camera_<键> = HDF5 相机名）。"""
    return list(load_sim_config()["cameras"].keys())


def look_at_quat_opengl(eye, target, up=(0.0, 0.0, 1.0)) -> tuple:
    """世界系 look-at → 四元数 (w,x,y,z)，OpenGL 约定（前向 -Z，上 +Y）。"""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    up_v = np.cross(back, right)
    R = np.stack([right, up_v, back], axis=1)        # 列 = 相机轴在世界系
    # 旋转矩阵 → 四元数 (w,x,y,z)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
        q = [0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        w = (R[k, j] - R[j, k]) / s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        x, y, z = q
    return (float(w), float(x), float(y), float(z))


# --------------------------------------------------------------------------- #
# 场景
# --------------------------------------------------------------------------- #

@configclass
class HangCupSceneCfg(InteractiveSceneCfg):
    """双臂工位 + 桌 + 杯 + 杯架钉。robot/camera 由
    HangCupEnvCfg.__post_init__ 从注册表与 YAML 填充（相机按 YAML
    cameras 键动态挂成员 camera_<键>）。"""

    # 地面
    # 地面：不用默认网格地面（其自带蓝色反照率，tint 盖不住），
    # 换浅灰大平板，视觉对齐 real 的浅色背景
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.05)),
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.72, 0.72)),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=700.0),
    )

    # 桌面（静态碰撞体）：1.3 × 0.9 × 0.05，顶面 z=TABLE_Z
    # 颜色对齐 real 0520 批：浅色桌沿 + 蓝灰台垫（见下方 mat）
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.3, 0.9, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.80, 0.82)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, TABLE_Z - 0.025)),
    )

    # 蓝灰台垫（real 桌上那块大垫子；静态薄片，只改视觉+接触摩擦面）
    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableMat",
        spawn=sim_utils.CuboidCfg(
            size=(1.1, 0.75, 0.004),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.36, 0.42)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, TABLE_Z + 0.002)),
    )

    # 杯子（动力刚体）：assets/cup_tube.usda（gen_cup_usda.py 生成）。
    # 视觉/碰撞解耦：视觉=细管网格；碰撞=12 box 环（内壁 r=0.015/外壁
    # r=0.021/高 0.051，内外壁+顶沿全覆盖）+ 原生圆柱底——2026-09-18
    # 用户两轮确诊：v1 实心杯让钉进不了腔（hung 物理不可达）；v2 非水密
    # 网格凸分解烹饪错误，内壁无碰撞抓不住且薄壁穿模。原生图元接触是
    # PhysX 最稳路径。2026-09-20 用户裁决：全尺寸 ×0.6（骑跨余量
    # 5mm→19mm/侧，破"蹭而不抓"），奖励常量同步迁移（静止杯心
    # 0.795→0.777）。
    # 质量 0.2kg（2026-09-19 穿模彻查裁决，烘焙进资产）：0.05kg 时位置
    # 驱动夹爪（等效 ∞ 质量）对杯质量比 ~∞:1，棱接触硬推下 PhysX 给
    # 3~6mm 穿透；0.2kg + min_pos_iter=8 实测 5.91mm→0.44mm。比真纸杯
    # 重是已知失真，换取接触稳定（用户裁决）。尺寸 ×0.6 时质量不缩——
    # 质量是接触稳定性参数，小杯同质量更稳。
    # 注意：碰撞体是 /Cup 的子 prim，接触传感器 filter 必须同时含
    # "/Cup" 与 "/Cup/.*"（见下方传感器定义）。
    cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_CONFIG_PATH.parent.parent / "assets" / "cup_tube.usda"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0),  # 慢速推出，防轻杯被弹飞
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.55, 0.38)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.1, TABLE_Z + 0.027)),   # 杯半高 0.027（×0.6 后）
    )

    # 杯架（对齐 real 白色线架的抽象版）：底板 + 双立柱 + 横梁 + 挂钉。
    # 位置在近侧右侧（相机在 -y 对向看过来时呈画面右下，同 real）。
    # 仅 "peg"（挂钉）参与奖励/终止判定，其余为装饰静态件。
    rack_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RackBase",
        spawn=sim_utils.CuboidCfg(
            size=(0.16, 0.10, 0.008),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.92)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(RACK_X, RACK_Y, TABLE_Z + 0.004)),
    )
    rack_post_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RackPostLeft",
        spawn=sim_utils.CylinderCfg(
            radius=0.004, height=0.11,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.92)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(RACK_X - 0.07, RACK_Y + 0.035, TABLE_Z + 0.055)),
    )
    rack_post_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RackPostRight",
        spawn=sim_utils.CylinderCfg(
            radius=0.004, height=0.11,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.92)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(RACK_X + 0.07, RACK_Y + 0.035, TABLE_Z + 0.055)),
    )
    rack_bar = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RackBar",
        spawn=sim_utils.CylinderCfg(
            radius=0.004, height=0.14, axis="X",
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.92)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(RACK_X, RACK_Y + 0.035, TABLE_Z + 0.11)),
    )

    # 挂钉（运动学刚体：有 root_pos 数据供奖励读取，但不被推动）；
    # 杯挂上 = 杯腔罩住钉（2026-09-21 路线 A：c5 插入谓词——杯底 z 低于
    # 钉顶+3cm 且 xy 对准 4cm，见 mdp_sm.SM_PARAMS）
    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.CylinderCfg(
            radius=0.005, height=0.12,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.92)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(RACK_X, RACK_Y, TABLE_Z + 0.06)),
    )

    # 由 HangCupEnvCfg.__post_init__ 填充（MISSING 模式，同 IsaacLab 惯例）
    robot_left: ArticulationCfg = MISSING
    robot_right: ArticulationCfg = MISSING
    # 相机不声明静态成员：__post_init__ 按 YAML cameras 键动态 setattr
    # camera_<键>（InteractiveScene 遍历 cfg.__dict__ 建实体，动态成员安全）

    # 夹爪-杯接触力传感器（2026-09-18 裁决：夹紧前与夹紧过程中用真实
    # 接触力塑形，替代/补充闭爪位置代理）。filter 到 Cup：force_matrix_w
    # 只含指-杯接触分量。杯碰撞体是 /Cup 的子 prim（wall_XX/bottom），
    # filter 必须显式含子路径 "/Cup/.*"，否则接触发生在子 prim 上时
    # 匹配不到、力恒 0。
    contact_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotLeft/panda_.*finger",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup", "{ENV_REGEX_NS}/Cup/.*"],
    )
    contact_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotRight/panda_.*finger",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup", "{ENV_REGEX_NS}/Cup/.*"],
    )

    # 惩罚用接触传感器（2026-09-18 用户裁决：①臂/爪碰桌面给负奖励；
    # ②除夹爪外臂体碰杯给负奖励——惩罚撞杯/撞桌的捷径策略，让
    # pregrasp 的"张爪对准包络位"成为相对最优）。
    # 桌传感器排除 link0（基座恒贴桌面，含了会常罚）与手指
    # （正确抓握时指尖必然降到垫面附近，含了会惩罚正确动作）；
    # 杯传感器排除手指（抓握本来就要指-杯接触，由 grip_force 奖励）。
    contact_table_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotLeft/panda_(link[1-7]|hand)",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table",
                                "{ENV_REGEX_NS}/TableMat"],
    )
    contact_table_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotRight/panda_(link[1-7]|hand)",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table",
                                "{ENV_REGEX_NS}/TableMat"],
    )
    contact_body_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotLeft/panda_(link[0-7]|hand)",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup", "{ENV_REGEX_NS}/Cup/.*"],
    )
    contact_body_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotRight/panda_(link[0-7]|hand)",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup", "{ENV_REGEX_NS}/Cup/.*"],
    )

    # 自碰撞惩罚传感器（2026-09-20 用户裁决）：
    # ① 单臂自碰（contact_self_*）：prim 与 filter 同 glob——force_matrix
    #    报"体 i × filter j"全部成对接触，自身对自身不产生接触项，关节
    #    直连相邻 link 由 PhysX 豁免；覆盖 link1-7+hand，**排除手指**
    #    （合法全闭/夹持时指-指、指-掌接触是正常动作，误罚会压闭爪）。
    #    物理前提：robots.py 已开 articulation self-collision。
    # ② 左右臂互碰（contact_arms_cross）：双臂是两个独立 articulation，
    #    互碰走普通接触，单侧挂传感器即可（接触对称）。
    contact_self_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotLeft/panda_(link[1-7]|hand)",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RobotLeft/panda_(link[1-7]|hand)"],
    )
    contact_self_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotRight/panda_(link[1-7]|hand)",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RobotRight/panda_(link[1-7]|hand)"],
    )
    contact_arms_cross = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotLeft/panda_(link[1-7]|hand)",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RobotRight/panda_(link[1-7]|hand)"],
    )

    # 指-臂碰撞惩罚传感器（2026-09-20 用户裁决：手指碰臂全罚，同侧对侧
    # 都要；但 ①指-桌 ②同爪指-指 不罚——前者是正确抓握必经，后者是
    # 合法全闭）。覆盖四种组合：
    #   同侧：本爪指 vs 本臂 link1-7+hand（指-hand 关节直连 PhysX 豁免）；
    #   对侧：本爪指 vs 对臂 link0-7+hand+双指（对爪指-指也算——任务中
    #   双爪指尖互顶无任何合法场景）。
    contact_fing_self_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotLeft/panda_.*finger",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RobotLeft/panda_(link[1-7]|hand)"],
    )
    contact_fing_self_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotRight/panda_.*finger",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RobotRight/panda_(link[1-7]|hand)"],
    )
    contact_cross_fing_l = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotLeft/panda_.*finger",
        update_period=0.0,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/RobotRight/panda_(link[0-7]|hand|.*finger)"],
    )
    contact_cross_fing_r = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotRight/panda_.*finger",
        update_period=0.0,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/RobotLeft/panda_(link[0-7]|hand|.*finger)"],
    )

    # 杯-桌接触力传感器（2026-09-21 路线 A 状态机新增：c3 离地谓词的
    # 支撑力判据——杯底 z 够高但桌力仍大 = 斜靠桌沿滑上，不算离地）。
    # prim 挂杯根 "/Cup"：UsdFileCfg spawn 不支持 activate_contact_sensors
    # （from_files 不消费该字段），PhysxContactReportAPI 已烘焙进
    # cup_tube.usda 根 prim（gen_cup_usda.py）；刚体根的 contact report
    # 覆盖其下全部碰撞子 prim。
    contact_cup_table = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table",
                                "{ENV_REGEX_NS}/TableMat"],
    )


# --------------------------------------------------------------------------- #
# MDP 配置
# --------------------------------------------------------------------------- #

@configclass
class ActionsCfg:
    # 2026-09-18 裁决：RL/录制的策略接口 = delta 增量动作（探索结构对齐
    # Isaac-Lift；绝对位 + std≈1 噪声的探索三年不收敛）。录制落盘仍记
    # processed 绝对目标——动作空间与数据契约解耦（用户裁决）。
    arm = mdp.DualArmDeltaPositionActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.proprio_pos)
        joint_vel = ObsTerm(func=mdp.proprio_vel)
        cup = ObsTerm(func=mdp.cup_pose)
        peg = ObsTerm(func=mdp.peg_pose)
        # last_action 钳 ±2（2026-09-21 lift 三次崩溃确诊：raw action 经
        # mu 随机游走可无界漂移，未钳观测直接灌炸 critic，详见
        # lift_env_cfg 同义项注释）。共享 PPO cfg，同步防护。
        last_action = ObsTerm(func=base_mdp.last_action, clip=(-2.0, 2.0))
        # 锁存阶段 one-hot（8 维，路线 A：奖励写、观测读同一份 buffer）
        z_latch = ObsTerm(func=mdp_sm.sm_z_latch_onehot)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CameraObsCfg(ObsGroup):
        """视觉观测组：不拼进 policy 向量（rsl_rl 只消费 policy/critic），
        供在线记录与未来像素策略使用（2026-09-13 裁决：视觉从第一版在环）。
        观测项取首路相机（front）；其余相机由录制脚本直读传感器。"""
        rgb = ObsTerm(func=mdp.camera_rgb, params={"name": "camera_front"})
        depth = ObsTerm(func=mdp.camera_depth, params={"name": "camera_front"})

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    camera: CameraObsCfg = CameraObsCfg()


@configclass
class RewardsCfg:
    # 2026-09-21 路线 A 重构（规格 refactor_task_mdp_with_state_machine.md）：
    # 14 项乘性门控链 → 谓词 c0–c6（滞回）+ 锁存阶段 z_latch + 首达奖金
    # （势函数差分，振荡刷分构造上无收益）+ 分段整形 + 时间罚 + 安全罚。
    # 旧链裁决①–⑯的除错逻辑全部内迁到谓词定义（见 mdp_sm.py 各谓词注释）：
    # 张爪要求→c0 开度；身体压杯扎营→c1 仅手指传感器；空捏/同向蹭→c2
    # 复用 _clamp 三条件；撞飞白嫖→锁存推进序+c3 桌力+success 准静态门；
    # 角色分裂→固定右臂操作。
    # 注意：RewardManager 对所有项 ×dt≈1/15 均匀缩放，项间比例不变。
    # 注意：本类不能有 RewardTermCfg 以外的类属性（manager 会全部当项遍历，
    # 连 _P 这样的别名也不行）——权重直写 mdp_sm.SM_PARAMS.xxx。
    first_visit = RewTerm(func=mdp_sm.sm_first_visit,
                          weight=mdp_sm.SM_PARAMS.w_stage)
    shape = RewTerm(func=mdp_sm.sm_shape, weight=1.0)   # λ 已在函数内分项
    time_pen = RewTerm(func=mdp_sm.sm_time, weight=mdp_sm.SM_PARAMS.p_time)
    safety_coll = RewTerm(func=mdp_sm.sm_safety_collision,
                          weight=mdp_sm.SM_PARAMS.p_coll)
    drop_pen = RewTerm(func=mdp_sm.sm_cup_dropped_pen,
                       weight=mdp_sm.SM_PARAMS.p_drop)
    success_bonus = RewTerm(func=mdp_sm.sm_success_bonus,
                            weight=mdp_sm.SM_PARAMS.b_success)
    action_rate = RewTerm(func=mdp.rew_action_rate, weight=-1e-3)
    # 保留的旧防碰约束（文档罚表未列，但裁决⑦的营地防病根：臂体压杯/
    # 砸桌捷径）。不终止，纯罚；若左臂学会乱动碰杯再升级为终止。
    pen_arm_table = RewTerm(func=mdp.pen_arm_table_contact, weight=-1.0)
    pen_body_cup = RewTerm(func=mdp.pen_body_cup_contact, weight=-1.0)


@configclass
class TerminationsCfg:
    # 路线 A：success/失败（真终止）与 max_steps 超时分离（文档 §7.3，
    # time_out=True 标记让 wrapper 对末步 value bootstrap 不混）。
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp_sm.sm_success)
    dropped = DoneTerm(func=mdp_sm.sm_drop_term)
    collision = DoneTerm(func=mdp_sm.sm_collision_term)
    # rack_tilt：杯架静态，接口留空（恒假，改动态刚体后启用）


@configclass
class EventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    # 状态机逐 env reset（文档 §7.2：不能只在整批 reset 时清）
    sm_reset = EventTerm(func=mdp_sm.sm_reset_state, mode="reset")
    # 杯摩擦材质（2026-09-18：此前未显式建模，吃 PhysX 默认 μ=0.5
    # average——位置驱动夹爪夹轻杯易挤出。μ_s=1.2/μ_d=1.0 对齐
    # 纸杯-橡胶指垫量级；与手指 0.5 合成 average≈0.85，2N 夹持下摩擦
    # 容量 ~3.4N ≫ 杯重 ~2N@0.2kg）。区间取点值 = 确定性设置，不随机化。
    set_cup_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("cup"),
                "static_friction_range": (1.2, 1.2),
                "dynamic_friction_range": (1.0, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1})
    randomize_cup = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("cup"),
                "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.1)},
                "velocity_range": {}})


# --------------------------------------------------------------------------- #
# EnvCfg
# --------------------------------------------------------------------------- #

@configclass
class HangCupEnvCfg(ManagerBasedRLEnvCfg):
    scene: HangCupSceneCfg = HangCupSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        cfg = load_sim_config()
        env_cfg = cfg["env"]

        self.decimation = env_cfg["decimation"]
        self.episode_length_s = env_cfg["episode_length_s"]
        self.sim.dt = env_cfg["sim_dt"]
        self.sim.render_interval = self.decimation
        self.scene.num_envs = env_cfg["num_envs"]

        # 2026-09-19 穿模彻查裁决（scripts/debug_penetration.py 实测）：
        # 默认 min_position_iteration_count=1 时，指尖楔在杯沿棱上被位置
        # 驱动持续硬推（22~69N），TGS 压不住 → 3~6mm 持续穿透（RL 策略
        # 未对准时的常态）。min_pos_iter 1→8 后同场景 5.91mm→0.15mm，
        # 零动力学副作用。vel 迭代同步 0→4（A/B 验证组合即 8/4）。
        self.sim.physx.min_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 4

        # 机器人：注册表构建双臂
        arms = get_spec(cfg["robot"]).build()
        self.scene.robot_left = arms["left"].replace(
            prim_path="{ENV_REGEX_NS}/RobotLeft")
        self.scene.robot_right = arms["right"].replace(
            prim_path="{ENV_REGEX_NS}/RobotRight")

        # 多相机：YAML cameras 每个键 → 场景成员 camera_<键>（2026-09-17
        # sawwam 裁决：2–3 路同录）。update_latest_camera_pose=True 是硬
        # 要求——录制脚本 set_world_poses 后要靠它刷新 pos_w/quat_w 缓冲
        # 才能读到新位姿写外参。
        for name, cam in cfg["cameras"].items():
            setattr(self.scene, f"camera_{name}", CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Camera_{name}",
                update_period=0.0,
                update_latest_camera_pose=True,
                width=cam["width"], height=cam["height"],
                data_types=cam["data_types"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=cam["focal_length_mm"],
                    clipping_range=(0.05, 10.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(cam["pos"]),
                    rot=look_at_quat_opengl(cam["pos"], cam["look_at"]),
                    convention="opengl",
                ),
            ))


def make_rl_cfg(num_envs: int | None = None) -> HangCupEnvCfg:
    """RL 训练用 env cfg：剥离全部相机（2026-09-17 裁决：阶段一纯状态
    策略，训练不渲染相机；像素策略是后续阶段二实验）。

    enable_cameras=False 启动时场景含相机会在 Camera._initialize_impl
    抛错，故必须 delattr 相机成员 + 置空 camera 观测组
    （ObservationManager 跳过 None 组，已核实）。
    """
    cfg = HangCupEnvCfg()
    cfg.scene.num_envs = int(num_envs or load_sim_config()["env"]["rl_num_envs"])
    for name in camera_names():
        delattr(cfg.scene, f"camera_{name}")
    cfg.observations.camera = None
    return cfg
