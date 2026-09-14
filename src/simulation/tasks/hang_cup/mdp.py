"""hang_cup 任务的 MDP 自定义件：16 维双臂绝对位置动作项 + 观测/奖励/终止。

动作契约（与 configs/data.yaml Franka 数据契约一致）：
    16 维 = [左臂 7 关节, 右臂 7 关节, 左夹爪, 右夹爪]，绝对位置指令；
    夹爪标量 ∈ [0, 0.04] m，广播到该手两个指关节（大=张开）。
    与 RoboMIND 数据 EE 语义（高=抓握）相反，重映射在数据对齐层处理。

观测：proprio 16 维（同契约顺序，rel 默认关节角）+ 关节速度 +
杯子/杯钉世界位姿 + last_action；相机在独立 obs group（不拼进 policy
向量，供记录/数据集落盘）。
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.sensors import Camera
from isaaclab.utils import configclass

_ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
_FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]


# --------------------------------------------------------------------------- #
# 动作项
# --------------------------------------------------------------------------- #

class DualArmAbsolutePositionAction(ActionTerm):
    """16 维绝对关节位置指令 → 两台 panda 实例。

    process: 直接裁剪到软关节限位（绝对位置，无 scale/offset）；
    apply: 左 7 + 右 7 写入臂关节目标，夹爪标量广播到两指关节。
    """

    _raw_actions: torch.Tensor
    _processed: torch.Tensor

    def __init__(self, cfg: "DualArmAbsolutePositionActionCfg",
                 env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._left: Articulation = env.scene[cfg.left_asset]
        self._right: Articulation = env.scene[cfg.right_asset]
        self._l_arm, _ = self._left.find_joints(_ARM_JOINTS, preserve_order=True)
        self._l_grip, _ = self._left.find_joints(_FINGER_JOINTS,
                                                 preserve_order=True)
        self._r_arm, _ = self._right.find_joints(_ARM_JOINTS, preserve_order=True)
        self._r_grip, _ = self._right.find_joints(_FINGER_JOINTS,
                                                  preserve_order=True)
        n = env.num_envs
        dev = env.device
        self._raw_actions = torch.zeros(n, 16, device=dev)
        self._processed = torch.zeros(n, 16, device=dev)
        # 软关节限位（裁剪范围）：臂 7 关节 + 夹爪行程 [0, travel]
        lo = self._left.data.soft_joint_pos_limits[0, self._l_arm, 0]
        hi = self._left.data.soft_joint_pos_limits[0, self._l_arm, 1]
        self._lo = torch.cat([lo, lo, torch.zeros(1, device=dev),
                              torch.zeros(1, device=dev)])[None]
        self._hi = torch.cat([hi, hi,
                              torch.full((1,), cfg.gripper_travel, device=dev),
                              torch.full((1,), cfg.gripper_travel, device=dev)])[None]

    @property
    def action_dim(self) -> int:
        return 16

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone()
        self._processed = torch.clamp(actions, self._lo, self._hi)

    def apply_actions(self) -> None:
        a = self._processed
        # 左臂 7 + 左爪（广播 2 指）
        self._left.set_joint_position_target(a[:, 0:7], joint_ids=self._l_arm)
        self._left.set_joint_position_target(
            a[:, 14:15].expand(-1, 2), joint_ids=self._l_grip)
        self._right.set_joint_position_target(a[:, 7:14], joint_ids=self._r_arm)
        self._right.set_joint_position_target(
            a[:, 15:16].expand(-1, 2), joint_ids=self._r_grip)


@configclass
class DualArmAbsolutePositionActionCfg(ActionTermCfg):
    class_type: type = DualArmAbsolutePositionAction
    asset_name: str = "robot_left"   # 基类 ActionTermCfg 必填字段；实际资产由
                                     # left_asset/right_asset 两个字段接管
    left_asset: str = "robot_left"
    right_asset: str = "robot_right"
    gripper_travel: float = 0.04


# --------------------------------------------------------------------------- #
# 观测
# --------------------------------------------------------------------------- #

def _proprio(env: ManagerBasedEnv, vel: bool) -> torch.Tensor:
    """16 维本体感：[左臂7, 右臂7, 左爪, 右爪]（vel=True 时速度版）。"""
    out = []
    for name in ("robot_left", "robot_right"):
        art: Articulation = env.scene[name]
        data = art.data
        q = data.joint_vel if vel else data.joint_pos
        arms, _ = art.find_joints(_ARM_JOINTS, preserve_order=True)
        grips, _ = art.find_joints(_FINGER_JOINTS, preserve_order=True)
        if not vel:
            default = data.default_joint_pos
            arm_part = q[:, arms] - default[:, arms]
        else:
            arm_part = q[:, arms]
        grip = q[:, grips].mean(dim=1, keepdim=True)
        out.append(torch.cat([arm_part, grip], dim=1))
    return torch.cat([out[0][:, :8], out[1][:, :8]], dim=1)


def proprio_pos(env: ManagerBasedEnv) -> torch.Tensor:
    return _proprio(env, vel=False)


def proprio_vel(env: ManagerBasedEnv) -> torch.Tensor:
    return _proprio(env, vel=True)


def cup_pose(env: ManagerBasedEnv) -> torch.Tensor:
    cup: RigidObject = env.scene["cup"]
    pos = cup.data.root_pos_w - env.scene.env_origins
    return torch.cat([pos, cup.data.root_quat_w], dim=1)


def peg_pose(env: ManagerBasedEnv) -> torch.Tensor:
    """目标钉顶点位姿（静态，直接从 prim 读出——奖励与观测共用）。"""
    peg: RigidObject = env.scene["peg"]
    pos = peg.data.root_pos_w - env.scene.env_origins
    return torch.cat([pos, peg.data.root_quat_w], dim=1)


def camera_rgb(env: ManagerBasedEnv, name: str = "camera_global") -> torch.Tensor:
    cam: Camera = env.scene.sensors[name]
    return cam.data.output["rgb"][..., :3].float() / 255.0     # (N,H,W,3)


def camera_depth(env: ManagerBasedEnv,
                 name: str = "camera_global") -> torch.Tensor:
    """深度观测，单位**毫米**、**1mm 整数量化**（2026-09-14 裁决：
    统一 RoboMIND 语义与精度——z-depth、mm、无效=0、量化粒度对齐
    uint16 mm，保证从训练模型视角两边数据本身一致）。

    取 distance_to_image_plane（z-depth，轴向距离），原始输出是米，
    ×1000 后四舍五入到整数 mm。容器仍为 float32（torch 观测需要），
    但取值只有整数。
    """
    cam: Camera = env.scene.sensors[name]
    d = cam.data.output["distance_to_image_plane"] * 1000.0    # (N,H,W,1) m→mm
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.round(d).clamp_(min=0.0, max=65535.0)


# --------------------------------------------------------------------------- #
# 奖励（最小分级：接近 → 抬起 → 挂到钉上）
# --------------------------------------------------------------------------- #

def _ee_pos(env: ManagerBasedEnv, side: str = "left") -> torch.Tensor:
    art: Articulation = env.scene[f"robot_{side}"]
    body_ids, _ = art.find_bodies("panda_hand")
    return art.data.body_pos_w[:, body_ids[0]] - env.scene.env_origins


def rew_reach_cup(env: ManagerBasedRLEnv) -> torch.Tensor:
    """左爪接近杯子（指数塑形）。"""
    cup = env.scene["cup"].data.root_pos_w - env.scene.env_origins
    d = torch.linalg.norm(_ee_pos(env, "left") - cup, dim=1)
    return torch.exp(-4.0 * d)


def rew_cup_lifted(env: ManagerBasedRLEnv, table_z: float = 0.75) -> torch.Tensor:
    cup_z = env.scene["cup"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp(cup_z - table_z, min=0.0)


def rew_cup_on_peg(env: ManagerBasedRLEnv) -> torch.Tensor:
    """杯子贴近钉顶点上方（挂上的塑形）。"""
    cup = env.scene["cup"].data.root_pos_w - env.scene.env_origins
    peg = env.scene["peg"].data.root_pos_w - env.scene.env_origins
    target = peg + torch.tensor([0.0, 0.0, 0.10], device=cup.device)
    d = torch.linalg.norm(cup - target, dim=1)
    return torch.exp(-6.0 * d)


def rew_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    a = env.action_manager.action
    prev = env.action_manager.prev_action
    return torch.sum((a - prev) ** 2, dim=1)


# --------------------------------------------------------------------------- #
# 终止
# --------------------------------------------------------------------------- #

def cup_hung(env: ManagerBasedRLEnv, tol: float = 0.05) -> torch.Tensor:
    cup = env.scene["cup"].data.root_pos_w - env.scene.env_origins
    peg = env.scene["peg"].data.root_pos_w - env.scene.env_origins
    target = peg + torch.tensor([0.0, 0.0, 0.10], device=cup.device)
    d = torch.linalg.norm(cup - target, dim=1)
    return d < tol


def cup_dropped(env: ManagerBasedRLEnv, table_z: float = 0.75) -> torch.Tensor:
    cup = env.scene["cup"].data.root_pos_w - env.scene.env_origins
    return cup[:, 2] < table_z - 0.15
