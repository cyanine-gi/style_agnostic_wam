# 双Franka「拿杯子放杯架」任务：奖励与观测规格（路线A版）

本文档供 coding agent 重写奖励模块使用。目标框架 Isaac Lab，算法 PPO，控制频率 15Hz。
所有阈值集中在第 10 节参数表，逻辑代码里不得出现硬编码数字。每个奖励项和观测项都要能独立开关、独立记录。

---

## 0. 任务定义

- 双 Franka，固定分工：
  - 右臂 R：抓杯、运输、插入杯架。
  - 左臂 L：episode 开始后移动到杯架旁的固定保持位，轻扶杯架直到 episode 结束。
- 对象：杯子 C，杯架有一个目标孔位，孔中心 p_slot，顶面高度 z_top。
- 成功：杯子底部进入孔位容差范围、速度低于阈值、持续 k_success 帧。成功即终止。释放动作单独作为一个阶段引导，不计入成功条件。
- episode 上限 max_steps，默认 400 步。

---

## 1. 设计约定

1. 允许历史记忆，但形式受限：全进程只有一个记忆变量——锁存阶段 `z_latch`（本 episode 到达过的最远阶段）。它是环境状态的一部分，奖励和观测都只读取它和当前物理状态。不允许引入第二个记忆变量。
2. 阶段首达给一次性奖金。放下杯子再抓起，z_latch 不回退，重复到达不重复发钱，振荡刷分没有收益。
3. 首达奖金取 `B_i = w_stage × i`，等价于在锁存阶段上做势函数差分，不改变最优策略，不引入局部最优。
4. 每个谓词由三部分组成：进入阈值、退出阈值（滞回）、持续帧判据。
5. 当前阶段 `z` 按链式定义：谓词组合异常时取最远连续前缀，不抛异常、不罚分，只记日志。
6. z_latch 以 one-hot 形式进观测，策略和 critic 都消费它。奖励计算和观测必须读写同一份 buffer，禁止两处各算一遍。

---

## 2. 符号

| 符号 | 含义 |
|---|---|
| p_gR, q_R, f_R | 右臂末端位置、夹爪开度（0 闭合，1 全开）、夹爪夹紧力 |
| p_C, v_C, z_C | 杯子位置、速度、底部高度 |
| p_slot, z_top | 目标孔中心（水平）、杯架顶面高度 |
| table_force_C | 杯子与桌面的接触力 |
| rack_tilt | 杯架倾角 |
| p_gL | 左臂末端位置 |

距离未注明时为 L2 范数，`||·||_xy` 为水平投影距离。所有位置为世界系。杯子和杯架的状态在仿真中取 ground truth，预留视觉估计接口。

---

## 3. 谓词

谓词是带滞回和持续帧判据的小状态机，骨架见 7.2。默认 k_enter=5 帧、k_exit=10 帧。

**c0 瞄准**：`||p_gR − p_C|| < d_aim_enter`（0.05 m）且 `q_R > 0.7`。退出距离 0.07 m，或开度低于 0.6。开度条件防止夹着杯子蹭瞄准分。

**c1 接触**：`f_R > f_touch_min`（2 N）或 `||p_gR − p_C|| < d_touch`（0.015 m）。退出：力低于 1 N 且距离大于 0.03 m。接触力取夹爪与杯子的接触对合力。

**c2 夹持**：`f_min_grasp < f_R < f_max_grasp`（8–40 N）且 `||p_C − p_gR|| < d_hold`（0.03 m）。f_min_grasp 必须大于空夹能达到的峰值力，先标定空夹一次，取 1.5 倍写入参数表。

**c3 离地**：`z_C > z_table + 0.05 m` 且 `table_force_C < 1 N`。支撑力判据用来排除杯子斜靠桌沿滑上去的情况。

**c4 上方**：`||p_C − p_slot||_xy < r_above`（0.08 m）且高度在 `z_top + [0.02, 0.25] m` 带内。上下界都卡，防止高空掠過和贴台面拖。

**c5 插入**：`||p_C − p_slot||_xy < r_insert`（先 0.04，收敛后收紧到 0.02）且 `z_C < z_top + 0.03 m`。阈值最紧的一条，退出阈值取 2 倍。

**c6 释放**：`f_R < 1 N` 且 c5 仍为真。可选，默认开启。

**success**：c5 持续 k_success 帧（30）且 `||v_C|| < 0.02 m/s`。

---

## 4. 阶段

```
z(s)     = max{ i ∈ [0..6] | c_0..c_i 全部为真 }        # 当前阶段，纯函数
z_latch  = max(z_latch, z)                               # 锁存，episode 内单调不减
```

异常谓词组合（如 c3 真 c2 假）时 z 自动落到最远连续前缀，不触发惩罚和终止，计数器记一次日志。

---

## 5. 奖励

每步：

```
r = R_first + R_shape − P_time − P_safety + R_terminal
```

**R_first（首达奖金，核心项，不可关）**
z_latch 从 i−1 升到 i 的那一步，发 `w_stage × i`（w_stage 默认 0.5）。用 z_latch 的增量检测，不要在谓词层重复检测。

**R_shape（前景整形，可关，默认开）**
只对当前 z 对应的下一阶段给引导，其他阶段不给：

| 当前 z | 整形项 |
|---|---|
| 0 | −λ · `||p_gR − p_C||` |
| 1 | −λ · `||p_gR − p_C||`，λ 减半 |
| 2 | +λ · clamp(`z_C − z_table`, 0, 0.05) |
| 3 | −λ · (`||p_C − p_slot||_xy` + `|z_C − (z_top + 0.10)|`) |
| 4 | −λ · (`||p_C − p_slot||_xy` + max(0, `z_C − z_top − 0.03`)) |
| 5 | 0（插入段不给整形，靠首达奖金提供动力） |
| 6 | −λ · `f_R`（引导松手） |

λ 默认 0.1，逐项可调 0。哪一项在被刷分，日志里能直接看出来，把它调 0 即可。

**P_time**：每步 −0.01，净量级不得超过单级首达奖金。

**P_safety**：

| 条件 | 罚分 | 终止 |
|---|---|---|
| 杯子掉到桌面以下 | −5 | 是 |
| rack_tilt > 10° | −5 | 是 |
| 关节越限、自碰撞、双臂碰撞 | −5 | 是 |
| 达到 max_steps | 0 | 是 |

**R_terminal**：success 时 +20，与终止同时发生。

掉杯罚分与成功奖金的比例要压住“故意摔杯重开”的策略：−5 对 +20 不够就提到 −10，不要靠调其他项硬压。

---

## 6. 观测

原有观测（图像、本体状态）之外追加 `z_latch` 的 7 维 one-hot。奖励模块每步把 z_latch 写入 env 级 buffer，观测项直接读它。c6 关闭时为 6 维。

一个已知风险：z 跌落后的恢复动作（滑落的杯子重新抓稳）发生在低位 z 条件下，但所需动作像高位 z。如果出现这种失败，补救办法是按 3 节的谓词原始值组一个低维向量拼进观测，替代单一 one-hot，先不改主结构。

---

## 7. 实现

### 7.1 奖励项注册

每个 RewardTerm 一个函数，cfg 从参数表读：

```python
def first_visit_bonus(env, std) -> torch.Tensor     # 5 节 R_first，含 z_latch 更新
def shape_reward(env, std) -> torch.Tensor          # R_shape，内部按 z 分支
def time_penalty(env, std) -> torch.Tensor
def safety_penalty(env, std) -> tuple[Tensor, Tensor]   # 罚分 + done 掩码
```

z_latch 的 buffer 挂在 env 上，形状 `(num_envs,)`，int8，初值 0。first_visit_bonus 负责更新它，观测项只读。禁止用模块级全局变量存 latch，向量化环境下这会写成跨 env 共享，表现为奖励突然集体异常。

### 7.2 谓词状态机

```python
class Predicate:
    def __init__(self, enter_fn, exit_fn, k_enter, k_exit):
        self.enter_fn, self.exit_fn = enter_fn, exit_fn
        self.k_enter, self.k_exit = k_enter, k_exit
        self._true = False
        self._run_true = 0
        self._run_false = 0

    def update(self, state) -> bool:
        cond = self.enter_fn(state) if not self._true else not self.exit_fn(state)
        if cond:
            self._run_true += 1; self._run_false = 0
        else:
            self._run_false += 1; self._run_true = 0
        if not self._true and self._run_true >= self.k_enter:
            self._true = True
        elif self._true and self._run_false >= self.k_exit:
            self._true = False
        return self._true
```

每个谓词每 env 一个实例。随 `reset_buf` 清零（Isaac Lab 逐 env 独立 reset，不能只在整批 reset 时清）。

### 7.3 每步顺序

1. 更新全部谓词实例
2. 算 z，更新 z_latch，算出 R_first
3. 检查 success 和失败终止条件，写 done、R_terminal、P_safety
4. 算 R_shape、P_time
5. 汇总 r，写观测 buffer，写日志

终止有两种：success/失败（真终止）和 max_steps（超时）。timeout 和 done 在 wrapper 里要分开传，最后一步的 value bootstrap 处理不能混。

---

## 8. 日志

每个 episode 记录：末态 z、max z_latch、success、episode 长度、终止原因（success / drop / rack_tip / collision / joint_limit / timeout）、分项奖励均值。另设一个谓词异常计数器，统计「c_i 真但 c_{i−1} 假」的出现次数。

checkpoint 评估跑不少于 100 个 episode，报两个数：成功率和部分分（max z_latch 均值除以 6）。部分分用于 checkpoint 预筛选，成功率用于最终比较。

---

## 9. 测试

1. 脚本走一遍理想轨迹，打印每步 z，必须单调走到 6，允许相邻阶段间重复，不允许无恢复的回跳。
2. 抓杯后松手，z 回落到 0 或 1，z_latch 不动，不终止不罚分。
3. 反复「夹起—放下」10 次：z_latch 不回退，累计 R_first 恒定。此条不过，不进训练。
4. hack 探针，各跑 50 步：空手悬停于杯上（c0 不得持续为真）；空夹满力（c2 不得为真）；横扫扇飞杯子（success 不得触发，c3 若持续为真超 1 秒则收紧支撑力判据）。
5. 把杯子放在 d_aim_enter 边界 ±1 mm 处抖动，谓词计数器不得高频翻转。
6. 随机 reset 100 次，首步全部谓词为假、z_latch 为 0。
7. 触发一次 max_steps 超时，确认 value bootstrap 处理正确（对照 wrapper 文档检查，不重算 value）。

---

## 10. 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| w_stage | 0.5 | 首达奖金系数，B_i = w_stage × i |
| B_success | 20.0 | 与掉杯罚分联动调 |
| P_drop / P_tip / P_coll | −5.0 | 出现摔杯重开行为时降到 −10 |
| λ | 0.1 | 整形权重，逐项独立 |
| d_aim_enter / exit | 0.05 / 0.07 m | 探索不足时放宽到 0.08 / 0.10 |
| r_above | 0.08 m | 卡在「上方」进不去时收紧到 0.05 |
| r_insert | 0.04 → 0.02 m | 最后一个调 |
| k_enter / k_exit | 5 / 10 帧 | 谓词抖振告警时加大 |
| k_success | 30 帧 | 误发 bonus 时加大到 50 |
| f_min_grasp | 8 N | > 1.5 × 空夹峰值力，先标定 |
| max_steps | 400 | 约为成功 episode 平均步数的 2 倍 |

调参顺序：谓词阈值正确性（测试 1、4）→ w_stage 量级 → 罚分与 bonus 平衡 → λ → r_insert。发现新的刷分路径，回谓词加判据，不加大惩罚硬压。

---

## 11. 验收

- 测试 1–7 全过。
- 训练曲线上部分分先于成功率上升，成功率随后跟上。
- 200 个评估 episode：成功率 ≥ 90%，部分分 ≥ 5.5 / 6。
- 谓词异常计数接近 0；不为 0 说明有未覆盖的状态组合，先查日志再调。