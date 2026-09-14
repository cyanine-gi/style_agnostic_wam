"""机器人注册表：任务与具体机器人解耦（2026-09-13 裁决：双臂起步，
且机器人必须可配置——任务代码只依赖本模块的注册键与统一动作契约，
换机器人 = 新增注册项 + 改 configs/simulation.yaml 的 robot 字段）。

统一动作/本体感契约（与 configs/data.yaml 的 Franka 数据契约一致）：
    16 维 = [左臂 7 关节, 右臂 7 关节, 左夹爪, 右夹爪]
    - 臂关节：绝对位置指令（rad），对应 panda_joint1..7；
    - 夹爪：标量，行程 [0, 0.04] m（手指关节目标，两个指关节广播同值；
      大=张开）。注意与 RoboMIND 数据的 EE 语义（高=抓握）相反，
      归一化/重映射在数据对齐层处理，不在本注册表。

新机器人注册项需提供：左右臂 ArticulationCfg（含安装位姿）、关节名映射、
EE link 名、home 关节角。
"""

from __future__ import annotations

from dataclasses import dataclass

# 注意：本模块不 import isaaclab（SimContext 启动前必须可导入）。
# ArticulationCfg 的构造推迟到 build() 调用时。


@dataclass(frozen=True)
class DualArmSpec:
    """双臂机器人规格（任务代码消费的唯一接口）。"""
    name: str
    arm_joints: int = 7                     # 每臂关节数
    gripper_travel: float = 0.04            # 夹爪行程（m）
    ee_link: str = "panda_hand"             # EE link 名（奖励/观测用）

    def build(self) -> dict:
        """返回 {"left": ArticulationCfg, "right": ArticulationCfg}。"""
        return _BUILDERS[self.name]()


def _build_franka_dual() -> dict:
    """双臂 Franka 工位：两台 panda 实例，安装于桌面左右两侧、面向中线。

    安装位姿为近似值（对齐 real 0520 批斜视机位中手臂左右伸入的布局），
    用 scripts/tune_camera.py 渲染核对后调整。
    home 姿态（2026-09-14 用户裁决"放平"）：不用 panda 默认的向上卷曲
    ready 姿态，改为水平伸向中线的平伸姿态，对齐真机数据起始帧。
    """
    from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
    from isaaclab.assets import ArticulationCfg

    # 平伸 home：j2 前倾 45°、j4 肘部展开、j6 腕部放平（手爪朝前水平）
    HOME_JOINT_POS = {
        "panda_joint1": 0.0, "panda_joint2": -0.785, "panda_joint3": 0.0,
        "panda_joint4": -2.356, "panda_joint5": 0.0, "panda_joint6": 1.571,
        "panda_joint7": 0.785, "panda_finger_joint.*": 0.04,
    }

    def arm(pos, yaw_deg):
        import math
        yaw = math.radians(yaw_deg)
        cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
        cfg.init_state.pos = pos
        cfg.init_state.rot = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        cfg.init_state.joint_pos = dict(HOME_JOINT_POS)
        return cfg

    return {
        # 桌面中心为原点系（scene 里桌子中心在 (0, 0)），左右对称安装
        "left": arm((-0.42, 0.15, 0.75), yaw_deg=-90.0),   # 面向 +x（中线）
        "right": arm((0.42, 0.15, 0.75), yaw_deg=90.0),    # 面向 -x（中线）
    }


_BUILDERS = {"franka_dual": _build_franka_dual}


def get_spec(name: str) -> DualArmSpec:
    if name not in _BUILDERS:
        raise KeyError(f"未知机器人 {name!r}，已注册：{sorted(_BUILDERS)}")
    return DualArmSpec(name=name)
