"""hang_cup 锁存阶段状态机（路线 A，2026-09-21 重构，规格见
src/simulation/refactor_task_mdp_with_state_machine.md）。

核心：7 个带滞回的小状态机谓词 c0–c6 + 锁存阶段 z_latch（episode 内单调
不减，全进程唯一记忆变量）。奖励 = 首达奖金（z_latch 增量 × w_stage，
势函数差分语义：重复到达不重复发钱，振荡刷分构造上无收益）+ 分段整形
+ 时间罚 + 安全罚（含终止）+ 成功终奖。

对文档的两处适配（2026-09-21 用户授权"细节可改、方向不变"）：
① 同时性前缀矛盾：文档 z=max{i|c0..ci 全真} 在"先张后闭"任务上不可达
  （c0 要求张爪 q>0.7，c2 起爪必闭 → 字面实现 z 永远过不了 1）。改为
  **锁存推进**：谓词 c_i 只在 z_latch==i 时参与推进判定，满足即 +1；
  z_latch 不回退。z_latch ∈ [0..7] = 已完成谓词数（文档 z∈[0..6] 等价
  于平移一位；one-hot 相应为 8 维）。谓词异常（c_j 为真但 latch<j）只
  计数记日志，不罚分不终止。
② 距离度量用**指尖中点**（mdp._tip_mid）而非 hand 原点——hand 原点距
  正确抓取位恒 ~10cm（v6 扎营确诊），文档 d_aim=5cm 在 hand 原点下
  物理不可达。

每步幂等：termination_manager 在 reward_manager **之前** compute
（manager_based_rl_env.py step 顺序），终止项先触发 update_stage_machine，
奖励/观测/脚本侧的同步重复调用由 common_step_counter 守卫直接吃缓存。
状态挂 env（禁止模块级全局——向量化环境下会跨 env 串扰）。

历史除错逻辑迁移对照（详见各谓词注释）：张爪要求→c0 开度条件；身体压杯
扎营→c1 只用手指传感器；空捏/指-指/同向蹭→c2 复用 _clamp 三条件；
撞飞白嫖→latch 推进序（先过 c2 才能到 c3）+ c3 桌力判据 + success
准静态门；角色分裂→固定右臂操作（文档分工，替代近手门控）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from . import mdp


# --------------------------------------------------------------------------- #
# 参数表（文档 §10：逻辑代码零硬编码，全集中于此；逐项可调 0/开关）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SMParams:
    # 奖励量级（RewardTermCfg weight 取这里；RewardManager 会 ×dt≈1/15，
    # 对所有项均匀缩放，项间比例不变——沿用旧 rew_hung_bonus 约定）
    w_stage: float = 0.5          # 首达奖金系数：到达第 i 级发 w_stage×i
    b_success: float = 20.0       # 成功终奖（与掉杯罚分联动调）
    p_time: float = -0.01         # 每步时间罚
    p_coll: float = -5.0          # 碰撞罚（自碰/指臂/双臂互碰）
    p_drop: float = -5.0          # 掉杯罚
    # 分段整形 λ（按下标 = 当前 latch 0..7；latch5 插入段=0 靠首达奖金驱动）
    # 2026-09-21（第三轮 iter 1281 c0 零点亮确诊）：λ=0.1 的入口势对 PPO
    # 噪声太弱（有效 ~0.007/步），c0 发现是高方差事件（第二轮 398 轮点亮
    # 是幸运分支）。stage-0/1 加强到 0.3 压探索方差；扎营无风险——
    # time_pen 使任何滞留净负，且势函数上限 0.3×0.7m/15≈0.014/步 仍
    # 小于单级首达奖金（0.5/15≈0.033）。
    # 2026-09-21（第四轮 iter 1130 diag 确诊）：夹后悬停 2.6cm 扎营——
    # 抬升段（latch 3）λ 同样 0.1→0.3，c3 的 5cm 阈值需要强垂直梯度。
    lam: tuple = (0.3, 0.3, 0.1, 0.3, 0.1, 0.0, 0.1, 0.0)
    # c0 瞄准（指尖中点→杯心距离 + 张爪）
    # 2026-09-21（iter 832 探索不足确诊）：5cm 球 + 5 帧驻留 + 开度>0.7
    # 三重窗口，40M 控制步探索零命中 → 按文档 §10 放宽到 0.08/0.10
    d_aim_enter: float = 0.08
    d_aim_exit: float = 0.10
    q_open_enter: float = 0.7     # 开度/行程
    q_open_exit: float = 0.6
    gripper_travel: float = 0.04
    # c1 接触（指-杯合力 or 极近）
    # 2026-09-21（iter 2325 c1 零锁存确诊）：d<1.5cm 分支从杯外物理不可达
    #（杯外壁 r=2.1cm，指尖贴壁时中点仍距杯心 2-3cm）；f>2N×5帧 对滑动中
    # 的 0.2kg 杯太苛刻（轻戳 1-3 帧 <2N 全被 k_enter 滤掉，62M 步零命中）。
    # 放宽：f 2.0→1.0/退出 0.5，d_enter 0.015→0.025（贴壁几何可达）。
    f_touch_enter: float = 1.0
    f_touch_exit: float = 0.5
    d_touch_enter: float = 0.025
    d_touch_exit: float = 0.03
    # c2 夹持（复用 _clamp 三条件；f_min 适配说明见 SM 文档/计划：
    # 指-杯专用传感器 + min_gap 使空捏/指-指结构上力读数为 0，无需
    # 文档的"空夹峰值×1.5"标定；f_max 上限防硬顶楔持刷夹持）
    f_min_grasp: float = 0.3
    f_max_grasp: float = 40.0
    cos_max: float = 0.5
    min_gap: float = 0.005
    # 几何常量（z_table 必须与 env_cfg.TABLE_Z 一致；杯半高为 ×0.6 后值）
    z_table: float = 0.75
    cup_half_h: float = 0.027
    peg_half_h: float = 0.06      # 钉高 0.12，root 居中 → z_top=root+0.06
    # c3 离地（杯底高度 + 杯-桌接触力排除斜靠滑上）
    z_air_enter: float = 0.05     # 杯底高于桌面
    z_air_exit: float = 0.03
    f_table_max: float = 1.0
    # c4 上方（钉 xy 容差 + 杯底相对钉顶的高度带）
    r_above_enter: float = 0.08
    r_above_exit: float = 0.10
    z_above_lo: float = 0.02
    z_above_hi: float = 0.25
    z_above_mid: float = 0.10     # latch=3 整形的目标悬停高度（杯底相对钉顶）
    z_above_lo_exit: float = 0.01
    z_above_hi_exit: float = 0.28
    # c5 插入（退出阈值取进入 2 倍滞回）
    r_insert_enter: float = 0.04
    r_insert_exit: float = 0.08
    z_insert_enter: float = 0.03  # 杯底 < z_top + 此值
    z_insert_exit: float = 0.06
    # c6 释放
    f_release: float = 1.0
    # 滞回帧数（15Hz：k_enter=5≈0.33s，k_exit=10≈0.67s，k_success=30=2s）
    k_enter: int = 5
    k_exit: int = 10
    k_success: int = 30
    # 成功准静态门（v7 教训：飞掠过目标区不算挂上）
    v_still: float = 0.02
    # 安全开关（碰撞即终止若压垮早期探索，置 False 降级为纯罚）
    coll_terminate: bool = True
    # episode 上限（步）：约为成功 episode 平均步数 2 倍
    max_steps: int = 400
    # 谓词数（c0..c6）
    n_pred: int = 7


SM_PARAMS = SMParams()


# --------------------------------------------------------------------------- #
# 状态（per-env buffer，挂 env；模块级只放不可变的 SM_PARAMS）
# --------------------------------------------------------------------------- #

class StageState:
    """全部 per-env 状态：谓词滞回计数器 + z_latch + success 计数。

    谓词下标 i ∈ [0..6] 对应 c_i；z_latch ∈ [0..7] 为已完成谓词数，
    谓词 c_i 仅在 z_latch==i 时参与推进（适配①）。
    """

    def __init__(self, env: ManagerBasedEnv, p: SMParams):
        n, dev = env.num_envs, env.device
        self.true = torch.zeros(n, p.n_pred, dtype=torch.bool, device=dev)
        self.run_true = torch.zeros(n, p.n_pred, dtype=torch.long, device=dev)
        self.run_false = torch.zeros(n, p.n_pred, dtype=torch.long, device=dev)
        self.z_latch = torch.zeros(n, dtype=torch.long, device=dev)
        self.success_run = torch.zeros(n, dtype=torch.long, device=dev)
        self.success_flag = torch.zeros(n, dtype=torch.bool, device=dev)
        # 本 episode 的成功快照：reset 事件把 success_flag 拷进来再清 live
        # 值——record_hang_cup.py 在 step 返回后（env 已自动 reset）读它。
        self.last_episode_success = torch.zeros(n, dtype=torch.bool,
                                                device=dev)
        self.anomaly_total = torch.zeros(n, dtype=torch.long, device=dev)
        # 每步幂等守卫与缓存
        self._last_step = -1
        self._latch_delta = torch.zeros(n, dtype=torch.float, device=dev)
        # 几何缓存（钉静态）：p_slot xy 与 z_top（env 原点系）
        self._slot_xy: torch.Tensor | None = None
        self._z_top: torch.Tensor | None = None


def _state(env: ManagerBasedEnv) -> StageState:
    st = getattr(env, "_sm_state", None)
    if st is None:
        st = StageState(env, SM_PARAMS)
        env._sm_state = st
    return st


def sm_reset_state(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """reset 事件：清谓词计数器/z_latch/success 计数（逐 env，文档 §7.2）。

    先把本 episode 的成功结果快照到 last_episode_success（录制脚本在
    自动 reset 之后读这个判成功），再清 live 状态。"""
    st = _state(env)
    st.last_episode_success[env_ids] = st.success_flag[env_ids]
    st.true[env_ids] = False
    st.run_true[env_ids] = 0
    st.run_false[env_ids] = 0
    st.z_latch[env_ids] = 0
    st.success_run[env_ids] = 0
    st.success_flag[env_ids] = False
    # _latch_delta/_last_step 不清：同步的终止步奖励仍要读缓存增量


# --------------------------------------------------------------------------- #
# 谓词原始条件（向量化，不含滞回；全部右臂——文档固定分工，替代近手门控）
# --------------------------------------------------------------------------- #

def _geom(env: ManagerBasedEnv, st: StageState, p: SMParams):
    """本步几何/力读数（env 原点系）。返回命名元组式 dict。"""
    cup = env.scene["cup"].data.root_pos_w - env.scene.env_origins
    if st._slot_xy is None:
        peg = env.scene["peg"].data.root_pos_w - env.scene.env_origins
        st._slot_xy = peg[:, :2].clone()
        st._z_top = peg[:, 2].clone() + p.peg_half_h
    tip = mdp._tip_mid(env, "right")
    d_tip = torch.linalg.norm(tip - cup, dim=1)
    openness = mdp._grip_open(env, "right") / p.gripper_travel
    f_cup = mdp._finger_cup_force(env, "right")          # 双指合力（N）
    clamp_ok, min_f = mdp._clamp(env, "right", f_min=p.f_min_grasp,
                                 cos_max=p.cos_max, min_gap=p.min_gap)
    z_bottom = cup[:, 2] - p.cup_half_h
    f_table = mdp._sensor_force_sum(env, "contact_cup_table")
    d_slot_xy = torch.linalg.norm(cup[:, :2] - st._slot_xy, dim=1)
    z_rel = z_bottom - st._z_top                          # 杯底相对钉顶
    v_cup = torch.linalg.norm(env.scene["cup"].data.root_lin_vel_w, dim=1)
    return dict(d_tip=d_tip, openness=openness, f_cup=f_cup,
                clamp_ok=clamp_ok.bool(), min_f=min_f, z_bottom=z_bottom,
                f_table=f_table, d_slot_xy=d_slot_xy, z_rel=z_rel,
                v_cup=v_cup)


def _raw_conditions(g: dict, st: StageState, p: SMParams) -> torch.Tensor:
    """c0..c6 的**进入**侧原始条件 (N,7)。退出侧在滞回更新里单独算。"""
    c = []
    # c0 瞄准：近且张爪（裁决⑧⑪ 迁移：接近必须张爪）
    c.append((g["d_tip"] < p.d_aim_enter) & (g["openness"] > p.q_open_enter))
    # c1 接触：指-杯力 or 极近（只用手指传感器——身体压杯不算，裁决⑦）
    c.append((g["f_cup"] > p.f_touch_enter) | (g["d_tip"] < p.d_touch_enter))
    # c2 夹持：_clamp 三条件 + 力上限（防硬顶楔持）
    c.append(g["clamp_ok"] & (g["min_f"] < p.f_max_grasp))
    # c3 离地：杯底够高且几乎不压桌（防斜靠桌沿滑上）
    c.append((g["z_bottom"] > p.z_table + p.z_air_enter)
             & (g["f_table"] < p.f_table_max))
    # c4 上方：钉 xy 容差 + 高度带（防高空飞掠/贴台拖）
    c.append((g["d_slot_xy"] < p.r_above_enter)
             & (g["z_rel"] > p.z_above_lo) & (g["z_rel"] < p.z_above_hi))
    # c5 插入：xy 对准 + 杯底压到钉顶附近（挂钉语义映射）
    c.append((g["d_slot_xy"] < p.r_insert_enter)
             & (g["z_rel"] < p.z_insert_enter))
    # c6 释放：指-杯力消失（c5 仍真由锁存推进序保证）
    c.append(g["f_cup"] < p.f_release)
    return torch.stack(c, dim=1)


def _exit_conditions(g: dict, st: StageState, p: SMParams) -> torch.Tensor:
    """c0..c6 的**退出**侧原始条件 (N,7)（滞回带宽与进入不同）。"""
    c = []
    c.append((g["d_tip"] > p.d_aim_exit) | (g["openness"] < p.q_open_exit))
    c.append((g["f_cup"] < p.f_touch_exit) & (g["d_tip"] > p.d_touch_exit))
    c.append(~(g["clamp_ok"] & (g["min_f"] < p.f_max_grasp)))
    c.append((g["z_bottom"] < p.z_table + p.z_air_exit)
             | (g["f_table"] > p.f_table_max))
    c.append((g["d_slot_xy"] > p.r_above_exit)
             | (g["z_rel"] < p.z_above_lo_exit)
             | (g["z_rel"] > p.z_above_hi_exit))
    c.append((g["d_slot_xy"] > p.r_insert_exit)
             | (g["z_rel"] > p.z_insert_exit))
    c.append(g["f_cup"] >= p.f_release)
    return torch.stack(c, dim=1)


# --------------------------------------------------------------------------- #
# 每步更新（幂等）
# --------------------------------------------------------------------------- #

def update_stage_machine(env: ManagerBasedRLEnv) -> StageState:
    """谓词滞回更新 → 锁存推进 → success 计数。每步幂等（common_step_counter
    守卫）：终止管理器先调用，奖励/观测吃缓存。"""
    st = _state(env)
    p = SM_PARAMS
    if st._last_step == env.common_step_counter:
        return st
    st._last_step = env.common_step_counter

    g = _geom(env, st, p)
    enter = _raw_conditions(g, st, p)
    exit_ = _exit_conditions(g, st, p)

    # 滞回小状态机（文档 §7.2 骨架的向量化版）
    cond = torch.where(st.true, ~exit_, enter)
    st.run_true = torch.where(cond, st.run_true + 1,
                              torch.zeros_like(st.run_true))
    st.run_false = torch.where(cond, torch.zeros_like(st.run_false),
                               st.run_false + 1)
    flip_on = (~st.true) & (st.run_true >= p.k_enter)
    flip_off = st.true & (st.run_false >= p.k_exit)
    st.true = (st.true | flip_on) & ~flip_off
    st.run_true = torch.where(flip_off, torch.zeros_like(st.run_true),
                              st.run_true)
    st.run_false = torch.where(flip_on, torch.zeros_like(st.run_false),
                               st.run_false)

    # 锁存推进：c_i 仅在 latch==i 时推进（可同步连跳多级）
    delta = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(p.n_pred):
        adv = (st.z_latch == i) & st.true[:, i]
        bonus = adv.float() * (i + 1)        # B = w_stage×i（i 从 1 起）
        delta += bonus
        st.z_latch = torch.where(adv, st.z_latch + 1, st.z_latch)
    st._latch_delta = delta

    # 谓词异常计数：c_j 为真但 latch<j（组合异常，只记日志不罚）
    idx = torch.arange(p.n_pred, device=env.device)
    anomaly = (st.true & (st.z_latch.unsqueeze(1) < idx.unsqueeze(0))
               ).any(dim=1)
    st.anomaly_total += anomaly.long()

    # success：c5 持续 k_success 帧且准静态（v7 准静态门迁移）
    # 2026-09-21 第五轮 iter ~1900 假成功修复：谓词 true 标志无门控（只有
    # latch 推进有前缀门控），而桌面上的杯子 z_rel=−0.12 天然满足 c5 高度
    # 条件——杯被撞到钉座 4cm 内静止 2s 即白嫖 +20 与成功终止（实测
    # success=0.2% 而 z_latch_max=1.0，链条不可能到达）。成功必须挂在
    # latch≥6（c5 经链条锁存）上，与文档"z 为前缀"的语义对齐。
    still = (g["v_cup"] < p.v_still)
    ok = st.true[:, 5] & still & (st.z_latch >= 6)
    st.success_run = torch.where(ok, st.success_run + 1,
                                 torch.zeros_like(st.success_run))
    st.success_flag |= st.success_run >= p.k_success

    # 日志（extras["log"] 在 _reset_idx 重建；这里 setdefault 持续写标量）
    log = env.extras.setdefault("log", {})
    log["SM/z_latch_mean"] = st.z_latch.float().mean().item()
    log["SM/z_latch_max"] = st.z_latch.max().item()
    log["SM/success"] = st.success_flag.float().mean().item()
    log["SM/anomaly_total"] = st.anomaly_total.float().mean().item()
    return st


# --------------------------------------------------------------------------- #
# 奖励项
# --------------------------------------------------------------------------- #

def sm_first_visit(env: ManagerBasedRLEnv) -> torch.Tensor:
    """首达奖金 R_first（核心项）：z_latch 增量 × 到达级数，权重 w_stage。
    势函数差分语义：重复到达不重复发钱，振荡刷分无收益（文档 §5）。"""
    return update_stage_machine(env)._latch_delta


def sm_shape(env: ManagerBasedRLEnv) -> torch.Tensor:
    """分段整形 R_shape：只引导下一阶段（文档 §5 表）。λ 逐项见 SMParams.lam，
    调 0 即关该项。返回的是**已含 λ 的值**（cfg weight 取 1.0）。"""
    st = update_stage_machine(env)
    p = SM_PARAMS
    g = _geom(env, st, p)
    lam = torch.tensor(p.lam, device=env.device).gather(
        0, st.z_latch.clamp(max=p.n_pred))
    d_tip = g["d_tip"]
    lift = (g["z_bottom"] - p.z_table).clamp(0.0, p.z_air_enter)
    to_slot = g["d_slot_xy"] + (g["z_rel"] - p.z_above_mid).abs()
    insert = g["d_slot_xy"] + (g["z_rel"] - p.z_insert_enter).clamp_min(0.0)
    # latch=k 表示 c0..c_{k-1} 已完成，整形只引导下一个谓词 c_k。
    # 2026-09-21 修正 off-by-one：旧表从 latch2 起整体错位一档（latch3 给了
    # 运输项而非抬升项），diag 确诊策略夹后悬在 c3 阈值下方扎营——垂直梯度
    # 被 xy 运输项稀释。现对齐文档 §5（文档 z = latch−1）。
    per_stage = torch.stack([
        -d_tip,                       # 0→c0 瞄准：趋近
        -d_tip,                       # 1→c1 接触：继续趋近（文档 z=0 全 λ）
        -0.5 * d_tip,                 # 2→c2 夹持：守在杯旁（文档 z=1 λ 减半）
        lift,                         # 3→c3 离地：抬升（诊断出的关键段，λ=0.3）
        -to_slot,                     # 4→c4 上方：运到钉上方
        -insert,                      # 5→c5 插入：λ=0 关闭，靠首达奖金驱动
        -g["f_cup"],                  # 6→c6 释放：引导松手
        torch.zeros_like(d_tip),      # 7：完成
    ], dim=1)
    val = per_stage.gather(1, st.z_latch.clamp(max=p.n_pred)
                           .unsqueeze(1)).squeeze(1)
    return lam * val


def sm_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    """时间罚：全 1，cfg weight = p_time（净量级 < 单级首达奖金）。"""
    update_stage_machine(env)
    return torch.ones(env.num_envs, device=env.device)


def sm_safety_collision(env: ManagerBasedRLEnv) -> torch.Tensor:
    """碰撞罚指示（自碰 + 指臂 + 双臂互碰；复用旧指示或）。cfg weight=p_coll。"""
    update_stage_machine(env)
    return torch.maximum(mdp.pen_self_collision(env),
                         mdp.pen_finger_arm(env))


def sm_cup_dropped_pen(env: ManagerBasedRLEnv) -> torch.Tensor:
    """掉杯罚指示（杯掉到桌面以下）。cfg weight=p_drop。"""
    update_stage_machine(env)
    return mdp.cup_dropped(env).float()


def sm_success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """成功终奖：本步 success_run 首次达 k_success 的那一步给 1
    （cfg weight=b_success）。注意 RewardManager ×dt 均匀缩放所有项。"""
    st = update_stage_machine(env)
    return (st.success_run == SM_PARAMS.k_success).float()


# --------------------------------------------------------------------------- #
# 观测项
# --------------------------------------------------------------------------- #

def sm_z_latch_onehot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """z_latch 的 8 维 one-hot（文档适配①：latch∈[0..7]）。奖励写、观测读
    同一份 buffer。"""
    st = update_stage_machine(env)
    out = torch.zeros(env.num_envs, SM_PARAMS.n_pred + 1, device=env.device)
    out.scatter_(1, st.z_latch.unsqueeze(1), 1.0)
    return out


# --------------------------------------------------------------------------- #
# 终止项
# --------------------------------------------------------------------------- #

def sm_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """成功终止：c5 持续 k_success 帧且准静态。成功即终止（文档 §0）。"""
    return update_stage_machine(env).success_flag


def sm_collision_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    """碰撞终止（自碰/指臂/双臂互碰）。coll_terminate=False 时降级为纯罚
    （恒假，风险开关见 SMParams）。"""
    update_stage_machine(env)
    if not SM_PARAMS.coll_terminate:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (mdp.pen_self_collision(env) + mdp.pen_finger_arm(env)).bool()


def sm_drop_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    """掉杯终止：杯掉到桌面以下（复用旧判据）。关节越限由 delta 动作项
    软限位 clip 在结构上覆盖（文档罚表第 3 行），不另设终止。"""
    update_stage_machine(env)
    return mdp.cup_dropped(env)


def rack_tilt_exceeded(env: ManagerBasedRLEnv,
                       max_deg: float = 10.0) -> torch.Tensor:
    """rack_tilt 谓词接口（杯架静态，2026-09-21 用户裁决：恒假留接口）。
    杯架改动态刚体后在这里实现倾角读取。"""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
