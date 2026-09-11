# GT 监督构造（深度刷写，v1，2026-09-11 定稿框架）

> 本文档约定"训练用深度监督目标如何离线构造"。
> 定位：预处理管线的可插拔组件；下游（dataloader / 损失 / 训练入口）
> 只消费统一产物契约（§2），**不感知用了哪种实现**。
> 上游依据：guideline.md §4.2、§5；消费契约见 dataloader.md §4。
> 设计纪律（2026-09-11 与项目负责人确认）：**多种实现封装在同一产物契约
> 后面；现阶段只定契约与两个实现；凡需训练实测才能拍板的细节一律列入
> §5 暂缓项，禁止提前设计。**

## 1. 为什么是多实现而不是单一流程

深度监督怎么构造才算"对"，取决于真实数据分布（洞的比例与空间分布、
传感器噪声形态、教师在两域的偏差模式），这些在 Stage 0 实测前无法预知。
因此刷写层按"同一契约、多种实现"组织：换实现 = 换 config 指向的刷写
脚本 + 重新落盘产物，dataloader / 损失 / 训练代码零改动。

同时，实现 #1 并非过渡品——它是 guideline §5 验收门强制要求的
mask-only 基线（本方案 AbsRel 须相对它显著改善），两个实现都要长期存在。

## 2. 统一产物契约（所有实现必须产出）

逐帧、逐相机一份记录，落盘 shard（格式随 dataloader 预处理统一，见
dataloader.md §6）。字段：

| 字段 | shape | 存储 | 说明 |
|---|---|---|---|
| `sensor_disp` | (64, 64) | uint16 + per-frame (min,max) | 传感器深度，disparity 空间，掩码感知降采样后 |
| `sensor_mask` | (64, 64) | uint8 | 有效性掩码，**唯一规则**：`isfinite & 量程内`（量程 `[已核实 2026-09-11，Franka]`：real (1, 5000)mm、sim (1, 10000)mm，0=洞与 65535 远平面哨兵均被排除），池化按有效比例重建（阈值默认 0.5，见 configs/model.yaml `pool_valid_thresh`） |
| `teacher_disp` | (64, 64) | uint16 + per-frame (min,max) | 教师稠密目标；实现 #1 下**缺省**（slot 为空），实现 #2 必填 |
| `stats` | — | sidecar json | per-frame 量化统计与对齐系数，供反归一化、可视化、损失侧使用 |

契约约束：

- 三个张量同分辨率（默认 64×64，掩码感知降采样在刷写侧完成）；
- **几何与 RGB 预处理严格一致**：刷写前对深度/掩码施加与方案 B'
  完全相同的填黑几何（等比缩放 + 上下填黑，patch 对齐，见
  `src/sawvla/data/image.py` 讨论记录），填黑区 `sensor_mask=0`——
  RGB 与深度监督逐像素对齐是本契约的硬要求（2026-09-11 二次裁决
  方案 B' 后新增）；
- 满分辨率原图仅评估/可视化用，是否随产物落盘由存储预算决定
  （默认不落，重读原 HDF5）；
- 任何实现**不得**在契约外增加训练侧需要知道的字段——新信息先进
  sidecar，被证实需要后再升契约。

## 3. 实现 #1：raw+mask（mask-only 基线）

- `sensor_disp/sensor_mask` 直接来自传感器：转 disparity → 有效性掩码 →
  掩码感知降采样到 64×64；
- `teacher_disp` 缺省；损失侧对应关闭教师项（λ_teacher=0），
  即 guideline §5 的 mask-only 基线配置；
- 用途：基线对照（Stage 0 验收门）、以及"教师到底带来多少增益"的
  消融臂。

**在线变体（已落地，2026-09-11 裁决）**：Stage 0 快速验证期不落盘，
`OnlineDisparitySupervision`（src/sawvla/data/supervision.py）在
dataloader 内在线完成同一条流水线（掩码 → disparity → B' 同几何填黑
降采样，patch_px=4，16:9 内容行 12:48），已作为 dataset 默认监督；
未来切离线产物只需在 dataset 装配处换掉该类，训练侧零改动。
scripts/train_stage0.py 即以本变体 + λ_teacher=0 跑通冒烟
（outputs/stage0_smoke/）。

## 4. 实现 #2：教师预刷 + 已知部分尺度统一

流程（逐帧）：

1. Depth Anything V2-L 离线推理（两域同一 checkpoint，518px，
   batch 8），输出相对 disparity；
2. **尺度统一**：在该帧 `sensor_mask` 有效像素上，对教师输出做
   scale-and-shift 拟合到传感器 disparity（初始版本用普通最小二乘；
   稳健化列入 §5）；
3. 对齐后的稠密目标 → 掩码感知降采样到 64×64 → uint16 量化落盘，
   拟合系数记入 sidecar；
4. `sensor_disp/sensor_mask` 照常产出（L_metric 仍只在传感器有效像素
   上计算）。

**对齐只做一次的分工约定**：实现 #2 已在预处理侧完成尺度统一，训练侧
L_teacher 即直接回归（恒等对齐）；`losses/depth.py::ssi_teacher` 的
损失侧对齐**保留作消融**，由 config 二选一，禁止两侧同时对齐。

## 5. 暂缓项（明确不设计，等 Stage 0/1 实测证据）

以下候选都在契约内预留了位置，但**是否做、怎么做一律待训练后裁决**：

1. **时序中值补背景**（guideline §4.2-3，潜在实现 #3）：静止片段界定
   标准、补入像素的 mask 语义（置 1 还是独立标志位）均未定；
   风险是洞与材质强相关，补错 = 引入系统性偏差；
2. 教师对齐的稳健化（稳健回归 / 截尾 / RANSAC）——先看普通最小二乘
   在两域各画多少残差再决定；
3. uint16 量化范围的更优取法（per-frame min/max 为默认；全局分位数
   待证据）；
4. `pool_valid_thresh` 是否需要在产物中保留更细粒度（如有效像素计数）
   以便训练侧调阈值；
5. 教师 checkpoint 更换 / 多教师集成的可能性。

## 6. 与损失项的消费关系（现状对应）

| 产物字段 | 消费者 | 说明 |
|---|---|---|
| `sensor_disp` + `sensor_mask` | `metric_nll`（L_metric） | 仅有效像素，σ 软降权在损失侧学 |
| `teacher_disp` | L_teacher | 实现 #2 下直接回归；`ssi_teacher` 为消融臂 |
| `sensor_mask` | `multiscale_grad`、评估掩码 | 训练掩码唯一规则（§2）；评估侧手工规则只允许出现在评估代码（§7.4） |
| `stats` | 反归一化、可视化、σ 诊断 | 不进损失 |
