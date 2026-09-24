# 杯-夹爪穿模彻查与修复（2026-09-19）

记录 hang_cup 任务"RL 抓取策略穿模严重到几乎不可用"的完整排查与修复过程。
所有结论均以当日从零实测为准（用户裁决：不信任任何旧 checkpoint 与旧
诊断结论）。诊断脚本与原始日志均可复跑，见文末。

## 症状

RL 策略抓杯时，屏幕上手指/手掌大范围陷入杯壁，穿模画面持续存在，
录制数据不可用。

## 方法论

1. **stage 实测，不猜配置**：所有"以为设置上了"的东西（碰撞 API、
   contact offset、摩擦材质、求解器参数）一律在加载后的 USD stage /
   PhysX view 上逐 prim 打印确认。
2. **几何定量穿模，不靠肉眼**：从 USD 提取夹爪碰撞凸包真实顶点，
   与杯壁 12 box 环做**双向**穿透测试（指顶点∈壁 box / 壁 box 角点
   ∈指凸包），逐控制步输出穿透深度（mm）并与接触力对时。
3. **同场景控制变量 A/B**：每个候选修复单独/组合跑同一场景，
   以最深穿透与持续事件数裁决。

## 根因（唯一复现出的机制）

**PhysX 顺从穿透（compliance penetration），不是碰撞体问题。**

指尖/指腹楔在**杯沿棱边**上，被位置驱动（PD target）持续硬推
（实测接触力 22~69N，瞬时可达 150N）。在

- `min_position_iteration_count=1`（IsaacLab 默认），
- 质量比 ≈ ∞ : 0.05kg（位置驱动连杆等效无穷质量 vs 轻杯），
- 棱/角接触（接触面积极小，局部压强极大）

三者叠加下，TGS 求解器压不住，给出 **3~6mm 的持续性穿透**，直到杯
被顶飞才消失。RL 策略未对准时的楔/压/砸动作反复进入该状态，即
"穿模非常严重"的观感来源。

**面贴面接触（正确抓取）穿透恒为 0**——穿模全部来自棱/角接触。

### 逐项排除的假设（全部实测否定）

| 假设 | 实测结果 |
|---|---|
| 杯碰撞体缺失 / 凸分解烹饪错误 | 12 box 环 + 圆柱底全部带 CollisionAPI、contactOffset=0.008 正确加载 |
| 夹爪碰撞 hull 残缺 | 完整：52 顶点 / 18 hull 顶点，convexHull 烹饪 |
| 视觉网格 ≠ 碰撞网格（恒定视觉穿模） | 指腹区视觉最多超出碰撞 0.59mm；杯侧视觉=碰撞。**视觉穿模 = 碰撞穿透本身** |
| 高速隧道效应（60Hz 时代的旧结论） | 120Hz + 双方 8mm offset 下，240 步暴力随机动作（\|a\|≤0.9）零穿模 |
| 接触传感器漏检 | 穿透全程有力读数（22~69N），是求解器压不住，不是漏检 |

## 修复（2026-09-19 用户裁决，已全部落地）

| # | 改动 | 位置 | 作用 |
|---|---|---|---|
| 1 | `min_position_iteration_count` 1→8，`min_velocity_iteration_count` 0→4 | `tasks/hang_cup/env_cfg.py` `__post_init__` | 主修复：同场景 5.91mm→0.15mm，零动力学副作用 |
| 2 | 杯质量 0.05→0.2kg **烘焙进资产** | `assets/gen_cup_usda.py`（重新生成 `cup_tube.usda`） | 缓解极端质量比；叠加后 →0.44mm |
| 3 | 夹爪驱动刚度 2e3→500，damping 1e2→25 | `robots.py` | 消"25N 恒定硬夹"的 0.8~2.2mm 持续穿透（前两项压不住这块）；峰值夹持力 ~12N 仍 ≫ 保持所需 ~2N |

**最终验证**（仓库配置直读，无 flag，`debug_penetration.py` 全流程）：

| 阶段 | 修复前 | 修复后 |
|---|---|---|
| P2 下降楔杯沿 | 5.91mm，17 步持续 | ≤0.85mm，4 步瞬时（臂压 100~150N） |
| A 闭爪 | 0.8~2.2mm 持续（25N 硬夹） | 0.00mm |
| B 抬升 | — | 0.00mm |
| S3 暴力随机 240 步 | — | 0 事件 |

实测无效项（不采纳）：`enable_stabilization`、夹爪刚度单独降（不配套
iter+质量时收益不稳）。

**残留与代价**：
- P2 的瞬时 sub-mm 来自臂部位置驱动（stiffness 400）砸杯沿的 ~150N
  冲击，PhysX 正常合规范围，视觉不可辨。
- 杯 0.2kg 重于真纸杯（~50g）是已知失真，换接触稳定（用户裁决）。
- **动力学已变（质量×4、夹爪刚度÷4、求解迭代×8），旧 RL 策略必须重训。**

## 次要发现

1. **奖励链曾被穿模白拿**：基线楔住穿透 4.3mm 时，双指 28/20N、
   cos=-0.99 → `_clamp`/`grip_force` 判定为"合法夹持"（45 步触发 5
   步）。修复后该形态穿透归零，但判据本身对"棱上硬挤"无免疫力，
   后续若改奖励需注意。
2. **接触传感器 filter 报错**（`expected 2, found 15`）：`/Cup` Xform
   与 `/Cup/visual` 也匹配进 filter，占恒 0 死槽位。功能正常
   （`filter_count=15`，`sum(dim=2)` 正确），可清理为 `/Cup/wall_*`
   + `/Cup/bottom` 两模式。
3. **工程陷阱（本机 IsaacSim 5.1 实测）**：
   - `root_physx_view.set_masses` 后端拒绝（"Failed to set rigid
     body masses in backend"）——质量只能烘焙进资产；
   - 机器人为 instanceable USD，`stage.Traverse()` 默认**不进实例
     代理**，指碰撞体整棵不可见——遍历必须带
     `Usd.TraverseInstanceProxies()`（曾导致穿模测量静默恒 0 一轮）。

## 复跑与产物

诊断脚本（均支持 `--render` 存特写帧图，需 GPU 显存余量）：

```bash
# 全流程：stage 碰撞检视 + 静态对账 + IK 操作流逐步入侵测量 + 暴力随机
conda run -n env_isaaclab python src/simulation/scripts/debug_penetration.py [--render] [--pin-cup]
# 楔住场景控制变量实验（--iters/--mass/--grip-stiffness/--stab）
conda run -n env_isaaclab python src/simulation/scripts/exp2_wedging.py --tag xxx
# 视觉网格 vs 碰撞凸包逐点对账
conda run -n env_isaaclab python src/simulation/scripts/exp3_visual_vs_collision.py
```

原始日志与帧图：`outputs/debug_penetration/`（`run_*.log` 为各次
A/B，`exp2_*.log` 为变体矩阵，`frames_*/` 为渲染帧）。
