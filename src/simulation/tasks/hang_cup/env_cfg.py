"""hang_cup 任务的环境装配：场景 + MDP 配置 + EnvCfg。

任务（复刻 RoboMIND hang_cup_on_cup_holder 的抽象版）：桌面左侧随机位置
放一个杯子，左臂抓起挂到杯架钉上。道具全部是 primitive 几何体
（2026-09-13 裁决：不追求 photo-realistic；视觉信号在环但写实度随意）。

相机：高置固定全局相机（cameras.global，pos + look_at 世界系），位姿从
configs/simulation.yaml 读取——近似 real 0520 批斜视机位，细调用
scripts/tune_camera.py。未来加腕部相机 = YAML cameras 下加条目 +
scene cfg 加成员。

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
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp as base_mdp

from simulation.robots import get_spec
from . import mdp

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "simulation.yaml"
TABLE_Z = 0.75          # 桌面高度（场景几何的唯一事实源）
RACK_X, RACK_Y = 0.25, -0.20   # 杯架中心（近侧右侧，奖励/终止经由 peg 引用）


def load_sim_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


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
    """双臂工位 + 桌 + 杯 + 杯架钉 + 高置全局相机。robot/camera 由
    HangCupEnvCfg.__post_init__ 从注册表与 YAML 填充。"""

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

    # 杯子（动力刚体）：圆柱 r=0.035 h=0.09，牛皮纸色对齐 real
    cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        spawn=sim_utils.CylinderCfg(
            radius=0.035, height=0.09,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.55, 0.38)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.1, TABLE_Z + 0.045)),
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
    # 杯挂上 = 杯身到达钉顶附近（rew_cup_on_peg / cup_hung 的目标点）
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
    camera_global: CameraCfg = MISSING


# --------------------------------------------------------------------------- #
# MDP 配置
# --------------------------------------------------------------------------- #

@configclass
class ActionsCfg:
    arm = mdp.DualArmAbsolutePositionActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.proprio_pos)
        joint_vel = ObsTerm(func=mdp.proprio_vel)
        cup = ObsTerm(func=mdp.cup_pose)
        peg = ObsTerm(func=mdp.peg_pose)
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CameraObsCfg(ObsGroup):
        """视觉观测组：不拼进 policy 向量（rsl_rl 只消费 policy/critic），
        供在线记录与未来像素策略使用（2026-09-13 裁决：视觉从第一版在环）。"""
        rgb = ObsTerm(func=mdp.camera_rgb)
        depth = ObsTerm(func=mdp.camera_depth)

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    camera: CameraObsCfg = CameraObsCfg()


@configclass
class RewardsCfg:
    reach = RewTerm(func=mdp.rew_reach_cup, weight=1.0)
    lift = RewTerm(func=mdp.rew_cup_lifted, weight=5.0)
    on_peg = RewTerm(func=mdp.rew_cup_on_peg, weight=10.0)
    action_rate = RewTerm(func=mdp.rew_action_rate, weight=-1e-3)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    hung = DoneTerm(func=mdp.cup_hung, params={"tol": 0.05})
    dropped = DoneTerm(func=mdp.cup_dropped)


@configclass
class EventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
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

        # 机器人：注册表构建双臂
        arms = get_spec(cfg["robot"]).build()
        self.scene.robot_left = arms["left"].replace(
            prim_path="{ENV_REGEX_NS}/RobotLeft")
        self.scene.robot_right = arms["right"].replace(
            prim_path="{ENV_REGEX_NS}/RobotRight")

        # 高置固定全局相机（pos + look_at，世界系）
        cam = cfg["cameras"]["global"]
        self.scene.camera_global = CameraCfg(
            prim_path="{ENV_REGEX_NS}/GlobalCamera",
            update_period=0.0,
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
        )
