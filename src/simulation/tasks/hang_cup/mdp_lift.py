"""lift_cup 任务的奖励/终止：IsaacLab 原生 Lift-Cube 配方（向官方对齐版）。

2026-09-21 用户裁决：放下状态机路线（mdp_sm.py 保留不删），回到
Isaac-Lift-Cube-Franka 原生配方——dense reach（tanh 核）+ lift 主项
+ 小动作罚 + 掉杯终止。无阶段门控、无锁存、无滞回、无记忆变量。
场景/动作/观测/接触传感器全部复用 hang_cup（mdp.py / env_cfg.py）。

2026-09-23 用户裁决：血泪补丁（线性 lift 坡、_tip_mid 手算测距、
固定 −1e-3 动作罚、64 env、3e-4 lr）表现不如官方，逐项回退对齐：
  reaching_object（1−tanh(d/0.1)，w=1.0，ee_frame 测距）
      → rew_reach：官方 object_ee_distance 同款——FrameTransformer
      指尖中心 → 杯心（杯 root 即几何/碰撞中心，同官方 cube root
      定义，用户裁决①），指尖中心与抓取位重合，距离可归零
      （v6 _tip_mid 教训的前提是 hand 原点偏 ~10cm；官方 ee_frame
      带 0.1034 工具偏移后不存在该问题）。
  lifting_object（阶跃 z>h → 1，w=15）
      → rew_lift：杯心高过静置中心 minimal_height（默认 5cm，
      用户裁决②）阶跃给钱，与官方一致。
  action_rate_l2 / joint_vel_l2（−1e-4 起步 + 课程 10000 步升 −1e-1）
      → 权重与课程在 lift_env_cfg.RewardsCfg/CurriculumCfg
      （用户裁决③；joint_vel 对 robot_left，左臂干活——右臂已冻结）。
保留两条本场景实测必要的硬约束（原生配方没有，裁决⑦的防病根）：
  pen_arm_table（臂碰桌 −1）、pen_body_cup（臂体碰杯 −1），纯罚不终止。
成功终止：杯底 > 桌面 + h_target 且非高速飞掠（|v|<0.5m/s 一行门，
防拍飞杯越过阈值骗 +10——v7 教训的极简版）。官方无成功终止，
此为举杯子任务的成功收口，保留。

注意：RewardManager 对所有项 ×dt≈1/15 均匀缩放，项间比例不变。
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from . import mdp

TABLE_Z = 0.75        # 与 env_cfg.TABLE_Z 一致（场景几何唯一事实源）
CUP_HALF_H = 0.027    # 杯半高（×0.6 后，资产注释见 env_cfg.cup）
REST_Z = TABLE_Z + CUP_HALF_H   # 杯静置中心高度（root 即几何中心）


def _cup_pos(env) -> torch.Tensor:
    """杯心位置（env 原点系，有限化——保险丝见 mdp._finite）。"""
    return (mdp._finite(env.scene["cup"].data.root_pos_w, limit=5.0)
            - env.scene.env_origins)


def rew_reach(env: ManagerBasedRLEnv, std: float = 0.4) -> torch.Tensor:
    """趋近整形（对齐官方 object_ee_distance）：1 − tanh(d/std) ∈ (0,1)。

    d = ee_frame 指尖中心（干活左臂 panda_hand + 0.1034 偏移，
    单 target，target_pos_w 形状 (N,1,3)）→ 杯心。两边同取世界系
    （同官方 root_pos_w / target_pos_w 写法），杯 root 即几何/碰撞
    中心，指尖中心与抓取位重合，奖励峰可达。
    2026-09-23 左右纠正：右臂已退出任务（8 维左臂动作项，右臂锁
    home），双手 target + min 取距会产生梯度死区（锁死手距离为
    常量，更近时 min 对左臂动作梯度恒 0）——只保留干活的 ee_left 帧。"""
    ee_w = mdp._finite(env.scene["ee_frame"].data.target_pos_w[..., 0, :],
                       limit=5.0)
    cup_w = mdp._finite(env.scene["cup"].data.root_pos_w, limit=5.0)
    d = torch.linalg.norm(cup_w - ee_w, dim=1)
    return 1.0 - torch.tanh(mdp._finite(d) / std)


def rew_lift(env: ManagerBasedRLEnv, minimal_height: float = 0.06) -> torch.Tensor:
    """举起奖励（主项，对齐官方 lifting_object 阶跃）：
    杯心（几何中心，同官方 cube root）高过静置中心 minimal_height 给 1。"""
    h = _cup_pos(env)[:, 2] - REST_Z
    return (h > minimal_height).float()

def rew_lift2(env: ManagerBasedRLEnv, minimal_height: float = 0.20) -> torch.Tensor:
    """举起奖励（主项，对齐官方 lifting_object 阶跃）：
    杯心（几何中心，同官方 cube root）高过静置中心 minimal_height 给 1。"""
    h = _cup_pos(env)[:, 2] - REST_Z
    return (h > minimal_height).float()

def rew_success(env: ManagerBasedRLEnv, h_target: float = 0.15,
                v_max: float = 0.5) -> torch.Tensor:
    """成功一次性奖金指示：杯底过 h_target 且非飞掠。与终止项同判据。"""
    return _lifted(env, h_target, v_max).float()


def _lifted(env: ManagerBasedRLEnv, h_target: float,
            v_max: float) -> torch.Tensor:
    z_bottom = _cup_pos(env)[:, 2] - CUP_HALF_H
    v = torch.linalg.norm(
        mdp._finite(env.scene["cup"].data.root_lin_vel_w, limit=20.0), dim=1)
    return (z_bottom > TABLE_Z + h_target) & (v < v_max)


def lift_success(env: ManagerBasedRLEnv, h_target: float = 0.15,
                 v_max: float = 0.5) -> torch.Tensor:
    """成功终止：杯底稳定高过桌面 h_target。"""
    return _lifted(env, h_target, v_max)
