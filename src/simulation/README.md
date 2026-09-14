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

- 任务 `Saw-HangCup-FrankaDual-v0` 已注册：双臂 Franka（robot 注册表
  `robots.py`，可配置）+ 桌/杯/杯架钉 + 高置固定全局相机（近似 real 0520
  斜视机位，细调用 `scripts/tune_camera.py --pos/--look-at/--focal`）；
- 16 维绝对位置动作契约与 configs/data.yaml 一致（[L7,R7,gL,gR]），
  观测含 16 维 proprio + 杯/钉位姿 + RGB/depth（camera 独立 obs group，
  视觉从第一版在环）；
- RL 后端 rsl_rl，经 `rl/facade.py` 包装层隔离，未来可换；
- 单进程调试入口：`simulation.sim_context.SimContext`（SimulationApp 单例，
  先 enter 再 import `simulation.tasks`）；
- 合并环境（仿真+训练单进程）依赖清单：`src/simulation/requirements.txt`
  （新建 conda env 用，现有两个环境不动）。

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
- **动作/本体感**：16 维 [左臂7, 右臂7, 左爪, 右爪] 绝对位置（见
  robots.py 与 mdp.py 契约注释）；夹爪行程 [0,0.04] m 大=张开，与
  RoboMIND EE 语义（高=抓握）相反，重映射在数据对齐层。
