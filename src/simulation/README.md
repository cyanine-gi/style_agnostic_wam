# simulation：可控仿真环境（替代 RoboMIND 原始 sim 数据）

> 目的（2026-09-13 用户提出）：自建可控仿真环境，替代视角/相机机位
> 不可控的原始 RoboMIND2.0-Franka-sim 数据（其相机问题见
> docs/dataloader.md §12.1/§12.9）。本文档记录环境版本事实与设计裁决。

## 环境版本（2026-09-13 实测核实）

conda 环境 `env_isaaclab`（与主项目 `style_agnostic_wam` 环境独立）：

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.11.16 | |
| Isaac Sim | **5.1.0**（pip `isaacsim==5.1.0.0`） | |
| Isaac Lab | **v2.3.2**（commit 37ddf6268 "Bumps version to v2.3.2"） | 源码 editable 安装自 `/mnt/nvme4t/IsaacLab`（isaaclab 0.54.2 / assets 0.2.4 / tasks 0.11.12 / rl 0.4.7 / mimic 1.0.16） |
| torch | 2.7.0+cu128 | Isaac Sim 5.1 强制钉版，升级见下 |

注意：

- Isaac Sim 5.1 依赖钉版（装新包时禁止顶掉）：packaging==23.0、
  psutil==5.9.8、click==8.1.7、typing_extensions==4.12.2、torch==2.7.0；
  flatdict 需 `--no-build-isolation` + setuptools<81；
  stable-baselines3<2.8、wandb<0.23、onnx<1.19、ipython<9、starlette<0.46；
- 机器：i5-12600KF + RTX 4070 Ti SUPER 16GB，Ubuntu 24.04。

## 状态

冒烟通过（2026-09-14，`env_isaaclab`）：

```
python src/simulation/scripts/play_hang_cup.py --steps 60 --episodes 1 --random-steps 50
# -> outputs/sim_play/play_grid.png（全局相机 RGB + depth 网格）
```

排障档案：[PENETRATION_DEBUG.md](PENETRATION_DEBUG.md) —— 2026-09-19
杯-夹爪穿模彻查（根因=PhysX 顺从穿透；修复=求解迭代 8/4 + 杯 0.2kg +
夹爪刚度 500，含复跑脚本）。

- 任务 `Saw-HangCup-FrankaDual-v0` 已注册：双臂 Franka（robot 注册表
  `robots.py`，可配置）+ 桌/杯/杯架钉 + 多相机（2026-09-17 sawwam 裁决：
  YAML `cameras` 每个键 = 场景成员 `camera_<键>`，当前 front/left/right
  三路；front 近似 real 0520 斜视机位，细调用
  `scripts/tune_camera.py --camera <键> --pos/--look-at/--focal`）；
- 16 维绝对位置动作契约与 configs/data.yaml 一致（[L7,R7,gL,gR]），
  观测含 16 维 proprio + 杯/钉位姿 + RGB/depth（camera 独立 obs group，
  视觉从第一版在环）；
- RL 后端 rsl_rl，经 `rl/facade.py` 包装层隔离，未来可换；
- 单进程调试入口：`simulation.sim_context.SimContext`（SimulationApp 单例，
  先 enter 再 import `simulation.tasks`）；
- 合并环境（仿真+训练单进程）依赖清单：`src/simulation/requirements.txt`
  （新建 conda env 用，现有两个环境不动）。

## 数据集录制（2026-09-17 sawwam 裁决，guideline §6.3 硬前提）

管线：`scripts/train_rl.py`（阶段一纯状态 PPO，不渲染相机）→
`scripts/record_hang_cup.py`（策略驱动采集）。

- **相机随机化**：每集 reset 后按 YAML `randomize` 块独立采样各路
  pos（±0.15m）/look_at（±0.05m）/focal（16–20mm），**集内固定**；
  随机化只发生在录制脚本，RL 训练不渲染相机（策略是纯状态策略）。
- **K/T 逐集落盘**：精确 K（原始像素，未做 letterbox 折叠）+ T_base_cam
  （基座系 = 两臂基座中点、旋转单位阵，ROS 相机轴）写进 HDF5
  （`camera_intrinsics/<cam>/matrix`、`camera_extrinsics/<cam>`）；
  约定细则见 `recording.py` 模块文档——sawwam dataloader 以此为准。
- **落盘 schema**：仿 RoboMIND 子集（vlen JPEG/PNG + master/puppet 8 条
  曲线组 + metadata），现有 `sawvla.data.dataset.RoboMindDataset` 直接
  可读；夹爪录制时线性翻转对齐 RoboMIND 语义（高=抓握）且保连续
  （用户裁决，禁止二值化）；只落盘挂杯成功集（`--keep-failed` 另存）。

## 数据语义约定（与 RoboMIND 对齐）

- **深度（2026-09-14 裁决）**：统一 RoboMIND 语义与精度 = **z-depth
  （轴向）+ 单位毫米 + 无效=0 + 1mm 整数量化**（量化粒度对齐 RoboMIND
  的 uint16 mm，保证训练模型视角下两边数据本身一致）。仿真侧相机
  data_types 用 `distance_to_image_plane`（原始输出 float 米），读取
  边界统一 ×1000 并四舍五入到整数 mm：`mdp.camera_depth`（训练观测）
  与 `viz.read_depth_mm`（脚本）是唯一入口。剩余固有差异只有洞率
  （RoboMIND 有洞，仿真稠密无洞）。
- **深度可视化**：`viz.depth_to_rgb` 与数据侧
  `scripts/check_franka_dataset.depth_to_rgb` 同一逻辑（每帧 5–95 分位
  拉伸，红=远、蓝=近、无效=黑）——任何深度出图必须走它，禁止另起配色。
- **动作/本体感**：16 维 [左臂7, 右臂7, 左爪, 右爪]（见 robots.py 与
  mdp.py 契约注释）。2026-09-18 裁决：**策略接口与数据契约解耦**——
  策略侧动作项为增量式（`DualArmDeltaPositionAction`，clip [-1,1]×
  scale，臂 ±0.1 rad/步、爪 ±0.01 m/步；绝对位+std≈1 噪声的探索结构
  三轮不收敛），录制落盘记动作项 processed 的 16 维**绝对**目标；
  夹爪录制时翻转对齐 RoboMIND 语义（高=抓握）且保连续。
