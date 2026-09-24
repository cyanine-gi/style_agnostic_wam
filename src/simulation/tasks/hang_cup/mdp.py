"""hang_cup 任务的 MDP 自定义件：16 维双臂绝对位置动作项 + 观测/奖励/终止。

动作契约（与 configs/data.yaml Franka 数据契约一致）：
    16 维 = [左臂 7 关节, 右臂 7 关节, 左夹爪, 右夹爪]，夹爪行程
    [0, 0.04] m（大=张开，与 RoboMIND 相反，重映射在数据对齐层）。
    2026-09-18 裁决：**策略接口与数据契约解耦**——训练/录制的动作项是
    增量式 DualArmDeltaPositionAction（raw = 每步增量，clip [-1,1]×
    scale）；DualArmAbsolutePositionAction 保留备用。录制落盘一律记
    动作项 processed 的 16 维**绝对**目标，与策略接口无关。

观测：proprio 16 维（同契约顺序，rel 默认关节角）+ 关节速度 +
杯子/杯钉世界位姿 + last_action；相机在独立 obs group（不拼进 policy
向量，供记录/数据集落盘）。

奖励/终止：2026-09-21 路线 A 重构——任务奖励链整体迁往 mdp_sm.py
（锁存阶段状态机，规格见 refactor_task_mdp_with_state_machine.md）；
本模块保留动作项、观测、以及被状态机复用的几何/接触/夹持判据 helper
（_tip_mid/_grip_open/_finger_cup_force/_clamp/_quasistatic/
_sensor_force_sum/cup_dropped 与各 pen_* 碰撞指示）。
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.sensors import Camera, ContactSensor
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


class DualArmDeltaPositionAction(ActionTerm):
    """16 维增量关节位置指令（2026-09-18 裁决：RL 改用 delta 动作，录制
    与动作解耦——录制落盘记 processed **绝对**目标，16 维数据契约不变）。

    动机（v3–v5 三抡奖励塑形后确诊）：绝对位动作 + PPO 噪声 std≈1.0
    直接加在绝对目标上 ⇒ 臂 15Hz 大幅甩动、夹爪维被 [0,0.04] 裁剪成
    bang-bang 蜂鸣，"张爪包杯→闭拢保持"复合事件物理上无法被探索命中。
    delta 动作把噪声变成每步增量（积分平滑），对齐 Isaac-Lift 官方探索
    结构。raw clip [-1,1] × scale：臂 ±0.1 rad/步（15Hz ≈ 1.5 rad/s，
    贴 panda 物理速度上限），夹爪 ±0.01 m/步（全行程 4 步）。
    """

    def __init__(self, cfg: "DualArmDeltaPositionActionCfg",
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
        # 每维增量 scale：臂 14 维 + 夹爪 2 维
        self._scale = torch.tensor(
            [cfg.scale_arm] * 14 + [cfg.scale_grip] * 2, device=dev)[None]
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
        """16 维绝对目标（契约序 [L7,R7,gL,gR]）——录制落盘读这里。"""
        return self._processed

    def _current16(self) -> torch.Tensor:
        """当前关节状态按契约序拼成 16 维（夹爪 = 两指均值）。"""
        ql = self._left.data.joint_pos
        qr = self._right.data.joint_pos
        return torch.cat([
            ql[:, self._l_arm], qr[:, self._r_arm],
            ql[:, self._l_grip].mean(dim=1, keepdim=True),
            qr[:, self._r_grip].mean(dim=1, keepdim=True)], dim=1)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone()
        delta = torch.clamp(actions, -1.0, 1.0) * self._scale
        self._processed = torch.clamp(self._current16() + delta,
                                      self._lo, self._hi)

    def apply_actions(self) -> None:
        a = self._processed
        self._left.set_joint_position_target(a[:, 0:7], joint_ids=self._l_arm)
        self._left.set_joint_position_target(
            a[:, 14:15].expand(-1, 2), joint_ids=self._l_grip)
        self._right.set_joint_position_target(a[:, 7:14], joint_ids=self._r_arm)
        self._right.set_joint_position_target(
            a[:, 15:16].expand(-1, 2), joint_ids=self._r_grip)


@configclass
class DualArmDeltaPositionActionCfg(ActionTermCfg):
    class_type: type = DualArmDeltaPositionAction
    asset_name: str = "robot_left"
    left_asset: str = "robot_left"
    right_asset: str = "robot_right"
    gripper_travel: float = 0.04
    scale_arm: float = 0.1      # rad/步（15Hz ≈ 1.5 rad/s 上限）
    scale_grip: float = 0.01    # m/步（全行程 0.04 需 4 步）


class DualArmDeltaBinaryGripperAction(DualArmDeltaPositionAction):
    """臂 delta + 夹爪 binary（2026-09-21 用户裁决：对 NN 的表现必须对齐
    Isaac-Lift 官方 BinaryJointPositionActionCfg——官方 binary 无记忆
    无积分，正号=全开/负号=全闭，每步独立决策；而我们原 delta 夹爪的
    指令锚在测量位置上，噪声随机游走 + clamp 饱和 → bang-bang，且合爪
    夹到杯后测量值被杯挡住、"闭"成吸收态，提前合爪被结构性强化）。

    语义（与官方一致）：raw[:, 14:16] >= 0 → 全开（gripper_travel），
    < 0 → 全闭（0.0）。每步独立，不积分、不读当前位置。
    16 维契约不变：processed 仍是绝对目标 [L7,R7,gL,gR]，录制/观测/
    动作维数全部不受影响；网络结构与已训 ckpt 完全兼容。
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)   # 先按 delta 填全部 16 维
        # 覆写夹爪两维为 binary 绝对指令（官方约定：正=开，负=闭）
        self._processed[:, 14:16] = torch.where(
            actions[:, 14:16] >= 0.0,
            torch.full_like(actions[:, 14:16], self.cfg.gripper_travel),
            torch.zeros_like(actions[:, 14:16]))


@configclass
class DualArmDeltaBinaryGripperActionCfg(DualArmDeltaPositionActionCfg):
    class_type: type = DualArmDeltaBinaryGripperAction


class LeftArmDeltaBinaryGripperAction(ActionTerm):
    """8 维左臂动作项（2026-09-23 用户裁决"右手不参与"）：网络输出
    从 16 维减到 8 维，右臂彻底退出任务。

    动作契约 8 维 = [左臂 7 关节 delta（clip ±1 × scale_arm，
    与双臂版同速率上限），左爪 binary（>=0 全开 / <0 全闭，
    同官方 BinaryJointPositionActionCfg 语义，每步独立）]。
    右臂每个控制步显式下发默认 home 位姿 + 全开（不推动、不空转、
    proprio 变常量，消除空转噪音的源头）。

    processed_actions 为 8 维绝对目标 [L7, gL]（录制/落盘注意契约
    已从 16 维变更——与 configs/data.yaml 的 16 维 Franka 契约不再
    一致，跨任务落盘需自行对齐）。旧 16 维 ckpt 与观测（last_action
    16→8，总 55→47）均不兼容，需重训。
    """

    def __init__(self, cfg: "LeftArmDeltaBinaryGripperActionCfg",
                 env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._left: Articulation = env.scene[cfg.asset_name]
        self._right: Articulation = env.scene[cfg.right_asset]
        self._l_arm, _ = self._left.find_joints(_ARM_JOINTS, preserve_order=True)
        self._l_grip, _ = self._left.find_joints(_FINGER_JOINTS,
                                                 preserve_order=True)
        self._r_arm, _ = self._right.find_joints(_ARM_JOINTS, preserve_order=True)
        self._r_grip, _ = self._right.find_joints(_FINGER_JOINTS,
                                                  preserve_order=True)
        n = env.num_envs
        dev = env.device
        self._raw_actions = torch.zeros(n, 8, device=dev)
        self._processed = torch.zeros(n, 8, device=dev)
        self._scale = torch.full((1, 7), cfg.scale_arm, device=dev)
        lo = self._left.data.soft_joint_pos_limits[0, self._l_arm, 0]
        hi = self._left.data.soft_joint_pos_limits[0, self._l_arm, 1]
        self._lo = lo[None]
        self._hi = hi[None]

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """8 维绝对目标 [L7 臂目标, gL 爪目标]——录制落盘读这里。"""
        return self._processed

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone()
        delta = torch.clamp(actions[:, :7], -1.0, 1.0) * self._scale
        arm = torch.clamp(self._left.data.joint_pos[:, self._l_arm] + delta,
                          self._lo, self._hi)
        # 左爪 binary（官方约定：正=开，负=闭）
        grip = torch.where(
            actions[:, 7:8] >= 0.0,
            torch.full_like(actions[:, 7:8], self.cfg.gripper_travel),
            torch.zeros_like(actions[:, 7:8]))
        self._processed = torch.cat([arm, grip], dim=1)

    def apply_actions(self) -> None:
        a = self._processed
        self._left.set_joint_position_target(a[:, :7], joint_ids=self._l_arm)
        self._left.set_joint_position_target(
            a[:, 7:8].expand(-1, 2), joint_ids=self._l_grip)
        # 右臂锁 home + 全开：显式下发目标（不接管 = PD 失持掉落），
        # 与 reset 后的默认状态一致，proprio 保持常量。
        self._right.set_joint_position_target(
            self._right.data.default_joint_pos[:, self._r_arm],
            joint_ids=self._r_arm)
        self._right.set_joint_position_target(
            torch.full((a.shape[0], 2), self.cfg.gripper_travel, device=a.device),
            joint_ids=self._r_grip)


@configclass
class LeftArmDeltaBinaryGripperActionCfg(ActionTermCfg):
    class_type: type = LeftArmDeltaBinaryGripperAction
    asset_name: str = "robot_left"
    right_asset: str = "robot_right"
    gripper_travel: float = 0.04
    scale_arm: float = 0.1      # rad/步（15Hz ≈ 1.5 rad/s 上限，同双臂版）


# --------------------------------------------------------------------------- #
# 观测
# --------------------------------------------------------------------------- #

def _finite(x: torch.Tensor, limit: float = 1e6) -> torch.Tensor:
    """物理发散保险丝（2026-09-21 lift 训练两次 iter ~400 崩溃确诊）：
    PhysX 个别 env 求解器爆炸时 joint_vel/刚体位姿出 inf/NaN，经观测
    毒化 critic（本版 rsl_rl 的 empirical_normalization 已 deprecated
    静默无效，观测是原始值进网络！），value 输出→自举目标逐轮 ×100
    螺旋。注意 limit 必须取**物理合理范围**而非形式上的大数——±1e6
    的"有限值"对网络输入与 inf 同样致命。发散 env 靠 drop/timeout
    自然终止，垃圾数据温和化后无害通过。"""
    return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit)


def _proprio(env: ManagerBasedEnv, vel: bool) -> torch.Tensor:
    """16 维本体感：[左臂7, 右臂7, 左爪, 右爪]（vel=True 时速度版）。"""
    out = []
    for name in ("robot_left", "robot_right"):
        art: Articulation = env.scene[name]
        data = art.data
        q = _finite(data.joint_vel if vel else data.joint_pos,
                    limit=50.0 if vel else 6.5)  # panda 关节物理限位内
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
    pos = _finite(cup.data.root_pos_w, limit=5.0) - env.scene.env_origins
    return torch.cat([pos, _finite(cup.data.root_quat_w, limit=1.0)], dim=1)


def peg_pose(env: ManagerBasedEnv) -> torch.Tensor:
    """目标钉顶点位姿（静态，直接从 prim 读出——奖励与观测共用）。"""
    peg: RigidObject = env.scene["peg"]
    pos = peg.data.root_pos_w - env.scene.env_origins
    return torch.cat([pos, peg.data.root_quat_w], dim=1)


def camera_rgb(env: ManagerBasedEnv, name: str = "camera_front") -> torch.Tensor:
    cam: Camera = env.scene.sensors[name]
    return cam.data.output["rgb"][..., :3].float() / 255.0     # (N,H,W,3)


def camera_depth(env: ManagerBasedEnv,
                 name: str = "camera_front") -> torch.Tensor:
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
# 奖励 helper（2026-09-21 路线 A 重构后保留件：几何/接触/夹持判据与
# 惩罚指示，供 mdp_sm.py 状态机谓词与安全罚复用。旧奖励链
# reach/pregrasp/straddle/grip_hold/lifted/transport/on_peg/hung 已由
# 谓词 c0–c6 + 锁存首达奖金取代并删除）
# --------------------------------------------------------------------------- #

def _ee_pos(env: ManagerBasedEnv, side: str = "left") -> torch.Tensor:
    art: Articulation = env.scene[f"robot_{side}"]
    body_ids, _ = art.find_bodies("panda_hand")
    return art.data.body_pos_w[:, body_ids[0]] - env.scene.env_origins


def _body_cup_contact(env: ManagerBasedEnv,
                      force_thresh: float = 0.5) -> torch.Tensor:
    """除夹爪外臂体碰杯指示（0/1）——惩罚项与塑形门控共用（2026-09-20
    裁决：臂体碰杯时 reach/pregrasp 塑形清零，掐掉"臂压杯扎营"的
    正收益营地，v14 确诊营地 net≈+1.9/s 无限期白拿）。"""
    f = (_sensor_force_sum(env, "contact_body_left")
         + _sensor_force_sum(env, "contact_body_right"))
    return (f > force_thresh).float()


def _finger_positions(env: ManagerBasedEnv, side: str) -> torch.Tensor:
    """该臂两指端刚体位置 (N,2,3)（env 原点系，有限化——保险丝见 _finite）。"""
    art: Articulation = env.scene[f"robot_{side}"]
    ids, _ = art.find_bodies(["panda_leftfinger", "panda_rightfinger"],
                             preserve_order=True)
    return _finite(art.data.body_pos_w[:, ids],
                   limit=5.0) - env.scene.env_origins[:, None, :]


def _grip_open(env: ManagerBasedEnv, side: str) -> torch.Tensor:
    """该臂夹爪开度（两指关节均值，m；0.04=全张）。"""
    art: Articulation = env.scene[f"robot_{side}"]
    gi, _ = art.find_joints(_FINGER_JOINTS, preserve_order=True)
    return art.data.joint_pos[:, gi].mean(dim=1)


def _tip_mid(env: ManagerBasedEnv, side: str) -> torch.Tensor:
    """该臂双指尖中点（env 原点系，(N,3)）。"""
    return _finger_positions(env, side).mean(dim=1)


def _finger_cup_force_vec(env: ManagerBasedEnv, side: str) -> torch.Tensor:
    """该臂两指**各自**与杯的接触力向量（世界系，(N, 2, 3)，M 过滤维求和）。
    2026-09-19 确诊：filter 列表 ["/Cup", "/Cup/.*"] 中 "/Cup" 是 Xform
    无碰撞体恒 0，碰撞全在 wall/bottom 子 prim（"/Cup/*"）——此前只读
    filter 0 导致力读数恒 0：grip_force 收的是 F=0 高斯尾巴（指尖<6cm
    白拿 0.169），_held 结构性恒 0、lifted 永远不可能。必须 sum 全 filter。"""
    cs: ContactSensor = env.scene.sensors[f"contact_{side}"]
    return cs.data.force_matrix_w.sum(dim=2)


def _finger_cup_force(env: ManagerBasedEnv, side: str) -> torch.Tensor:
    """该臂两指-杯接触力幅值之和（N，标量版，诊断/兼容用）。"""
    return torch.linalg.norm(_finger_cup_force_vec(env, side), dim=-1).sum(dim=1)


def _clamp(env: ManagerBasedEnv, side: str, f_min: float = 0.3,
           cos_max: float = 0.5,
           min_gap: float = 0.005) -> tuple[torch.Tensor, torch.Tensor]:
    """夹持判据（2026-09-19 两轮用户裁决 + 2026-09-20 补第三条件）：
    ① 两指**各自**对杯力 > f_min；
    ② 力向量**不同向**（cos < 0.5）——v1 要求基本反向（cos < -0.3）被
       check_clamp 第二轮证伪：低位放置的稳态抓取是"一指压侧壁+一指压
       顶沿"的垂直捏持（cos≈0，22N/15N 稳定携带 +7.3cm），是合法抓取
       却被判负。放宽到 0.5 后只拒"双指同向蹭/拖"（cos→+1，v10
       exploit 形态）；
    ③ 两指开度 > min_gap（**没完全合上** ⇒ 指间确有物体，2026-09-20
       用户裁决：防空捏/指-指互碰被力条件误判——开自碰撞后全闭时
       指-指接触力真实存在，但那不是夹杯）。
    held 下游的 airborne 门保证 lifted 必须真离桌，机械上要求力闭合，
    同向拖拽物理上点不了火。返回 (夹持指示 float, min(f1,f2))。"""
    V = _finger_cup_force_vec(env, side)
    f = torch.linalg.norm(V, dim=-1)                     # (N, 2指)
    both = (f > f_min).all(dim=1)
    vn = V / f.clamp_min(1e-6).unsqueeze(-1)
    cos = (vn[:, 0] * vn[:, 1]).sum(dim=-1)
    gap_ok = _grip_open(env, side) > min_gap
    return (both & (cos < cos_max) & gap_ok).float(), f.min(dim=1).values


def _quasistatic(env: ManagerBasedEnv, vel_thresh: float = 0.2) -> torch.Tensor:
    """杯准静态标志（|v| < vel_thresh m/s）：防"高速飞掠目标区"被判成功
    ——飞掠不门控的话，录制的 success 集会被甩飞的集污染（v7 教训）。"""
    v = env.scene["cup"].data.root_lin_vel_w
    return (torch.linalg.norm(v, dim=1) < vel_thresh).float()


def _sensor_force_sum(env: ManagerBasedEnv, key: str) -> torch.Tensor:
    """接触传感器全部 body × 全部 filter 的合力幅值总和（N）。无接触 = 0。"""
    cs: ContactSensor = env.scene.sensors[key]
    return torch.linalg.norm(cs.data.force_matrix_w, dim=-1).sum(dim=(1, 2))


def pen_arm_table_contact(env: ManagerBasedRLEnv,
                          force_thresh: float = 0.5) -> torch.Tensor:
    """臂（link1-7+hand）碰桌面/台垫惩罚指示（2026-09-18 用户裁决）。
    指回 1 表示在碰，配负权重。排除 link0（基座恒贴桌面）与手指
    （正确抓握指尖必然降到垫面附近，误伤正确动作）。"""
    f = (_sensor_force_sum(env, "contact_table_left")
         + _sensor_force_sum(env, "contact_table_right"))
    return (f > force_thresh).float()


def pen_self_collision(env: ManagerBasedRLEnv,
                       force_thresh: float = 0.5) -> torch.Tensor:
    """自碰撞/臂间碰撞惩罚指示（2026-09-20 用户裁决）：单臂 link1-7+hand
    互碰（articulation self-collision，robots.py 开启）+ 左右臂互碰。
    排除手指——合法全闭/夹持时指-指、指-掌接触是正常动作。"""
    f = (_sensor_force_sum(env, "contact_self_left")
         + _sensor_force_sum(env, "contact_self_right")
         + _sensor_force_sum(env, "contact_arms_cross"))
    return (f > force_thresh).float()


def pen_finger_arm(env: ManagerBasedRLEnv,
                   force_thresh: float = 0.5) -> torch.Tensor:
    """指-臂碰撞惩罚指示（2026-09-20 用户裁决，与 pen_self 并列的专项）：
    手指碰**任何臂**都罚，同侧对侧全覆盖——本爪指 vs 本臂 link1-7+hand、
    本爪指 vs 对臂 link0-7+hand+双指。
    刻意不罚（用户裁决 1、2）：指-桌（正确抓握指尖必降到垫面附近）、
    同爪指-指（合法全闭）。"""
    f = (_sensor_force_sum(env, "contact_fing_self_left")
         + _sensor_force_sum(env, "contact_fing_self_right")
         + _sensor_force_sum(env, "contact_cross_fing_l")
         + _sensor_force_sum(env, "contact_cross_fing_r"))
    return (f > force_thresh).float()


def pen_body_cup_contact(env: ManagerBasedRLEnv,
                         force_thresh: float = 0.5) -> torch.Tensor:
    """除夹爪外臂体（link0-7+hand）碰杯惩罚指示（2026-09-18 用户裁决）
    ——压"手臂硬撞撞飞杯"的捷径；指-杯接触是合法抓握，不在此列。
    2026-09-20 起与塑形门控共用同一指示。"""
    return _body_cup_contact(env, force_thresh)


def rew_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    a = env.action_manager.action
    prev = env.action_manager.prev_action
    # 上限 64：|a|≤2 时 16 维最大 64——网络已被污染时出现巨值，
    # 钳住不再二次毒化 value target（保险丝同伴，见 _finite）
    return _finite(torch.sum((a - prev) ** 2, dim=1)).clamp(max=64.0)


# --------------------------------------------------------------------------- #
# 终止
# --------------------------------------------------------------------------- #

def cup_dropped(env: ManagerBasedRLEnv, table_z: float = 0.75) -> torch.Tensor:
    cup = env.scene["cup"].data.root_pos_w - env.scene.env_origins
    return cup[:, 2] < table_z - 0.15
