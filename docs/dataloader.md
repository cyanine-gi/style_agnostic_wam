# Dataloader 设计（v2，2026-09-11 Franka 切换后定稿）

> 本文档是数据加载层的唯一事实来源之一（与 `configs/data.yaml` 互为表里）。
> 上游依据：guideline.md §4（数据管线）、§5（深度监督）、§7（域隔离）。
> **不在本文档范围内**：深度监督目标的离线构造（教师推理、掩码生成、
> 监督目标降采样）——那是预处理侧的可插拔组件，设计见
> `docs/depth_gt_supervision.md`。本文档只约定加载层对其产物的消费方式。
>
> **v2 变更（2026-09-11，用户裁决）**：默认数据集从 Tienkung 切换为
> Franka（real = RoboMIND2.0-Franka-Part-1，sim = RoboMIND2.0-Franka-sim）——
> 同机器人、同任务、EE 同构的"完美对应"数据；Tienkung 全部归档，
> 后续有空再议。Tienkung 时代的相机/EE/动作质量问题随之作废，
> Franka 侧新问题登记在 §12。

## 1. 架构原则

**组件间无多余交互**：除 shape 契约（前后模块输入输出维度一致）外，
dataloader 不感知任何模型/训练内部实现——不知道 E/T/D 的存在，不知道
GRL 的 λ，不知道当前处于哪个 stage。stage 逻辑（用哪个视图、什么采样器、
k 取几）全部在 config 与训练入口侧组装。

推论：

- dataloader 只返回**统一样本 schema**（§4），消费者自取所需字段；
- 域均衡是**采样器**不是数据集（§3）；
- matched 对照集是**过滤视图**不是独立存储（§2.3）；
- 归一化统计是**预处理产物**，dataset 查表应用，不自行估计（§5）；
- **域差异（相机选择、深度哨兵规则、信号约定）封装在按域的 dataset
  子类里**（已裁决 2026-09-11，§2），共享读取代码但不共享域假设。

## 2. 数据集与三个逻辑视图

底层是**一个原子 Dataset 基类 + 两个按域子类**（共享同一份读取代码与
同一份落盘数据，域差异只存在于子类的默认装配参数里）：

- `FrankaRealDataset`：RoboMIND2.0-Franka-Part-1（300 success episodes /
  ~8.8 万帧 / 100GB，任务 hang_cup_on_cup_holder）；相机 camera_front
  （1280×720，**斜视**，见 §12.9），
  深度量程 (1, 5000)mm（0=洞、65535=远平面哨兵均被掩码排除）。
- `FrankaSimDataset`：RoboMIND2.0-Franka-sim（319 success episodes /
  ~5.7 万帧 / 48GB，任务 124-hang_cup_on_cup_holder）；相机 camera_front
  （1280×720，**近垂直顶视**，与 real 视角不一致，见 §12.9），
  深度量程 (1, 10000)mm。

三个逻辑视图：

### 2.1 视图 #1：真机全量

`FrankaRealDataset` 全部 episode 的逐帧记录。

### 2.2 视图 #2：仿真全量

`FrankaSimDataset` 全部 episode 的逐帧记录。

### 2.3 视图 #3：域对抗对照集（`MatchedView`）

**任务级 matched 子集**：`matched_tasks`（写在 `configs/data.yaml`，
禁止硬编码）在两个域内的全部帧。实现 = 同一个 Dataset 类 + task 白名单
过滤，**不复制数据**。

Franka 切换后的结构变化：两域各只有同一个对应任务
（hang_cup_on_cup_holder ↔ 124-hang_cup_on_cup_holder），
**全库即 matched**——GRL 对抗流与域探针直接在 #1/#2 全量上进行，
不再需要窄子集；#3 作为过滤机制保留，供未来多任务数据使用。

#3 的消费者有两个，且都依赖"任务内容在两侧对齐"这一性质：

1. **GRL 对抗流（Stage 2）**：判别器训练与 GRL 损失只在任务对齐的数据
   上进行，防止判别器靠任务内容作弊（guideline §7.2 的过杀失效模式）。
2. **域探针 / 评估**：域探针只在任务对齐数据上拟合与评估，否则测的是
   任务不是画风（guideline §7.3 口径）。

**配对粒度是任务级，不是帧级**：real/sim 的 episode 长度、轨迹不同，
不存在也不构造帧对帧配对；#3 只保证"同一任务在两个域都有样本"。

## 3. 域均衡采样器（不是数据集）

域帧量比 real:sim ≈ 8.8万 : 5.7万 ≈ 1.6:1（轻度不平衡），仍用
`DomainBalancedSampler` 解决：包在 #1+#2 的 ConcatDataset 外，每个 batch
强制 real/sim 各 50%。

- 它是采样逻辑，不引入新数据、不复制索引之外的东西；
- 各 stage 是否启用由 config 决定：
  - Stage 0：可选（教师监督两域对称，不均衡危害小，默认关，可开）；
  - Stage 1：**默认开**（防 T 被单域动力学主导）；
  - Stage 2：**开**（guideline 硬性要求 50/50）。

## 4. 统一样本 schema

所有视图返回同构字典。**k=0 时退化为单帧**（Stage 0），k>0 为 clip
（Stage 1/2）；shape 中的时间维长度一律 k+1（帧）与 k（动作/本体感）：

| 字段 | shape | dtype | 说明 |
|---|---|---|---|
| `rgb` | (k+1, 3, 224, 224) | float32 | ImageNet 归一化后（方案 B' 填黑，§5） |
| `depth` | (k+1, 64, 64) | float32 | 传感器深度，disparity 空间，掩码感知降采样后的监督目标 |
| `mask` | (k+1, 64, 64) | float32 | 有效性掩码（1=有效），唯一规则见 guideline §4.2-2 |
| `teacher` | (k+1, 64, 64) | float32 | 教师稠密伪深度（disparity），降采样后；在线路径（Stage 0）缺省 |
| `action` | (k, 16) | float32 | 逐步动作 `[已核实]`：双臂7关节+双夹爪 |
| `proprio` | (k, 16) | float32 | 本体感，同构 |
| `domain` | 标量 | int64 | 0=real, 1=sim |
| `task_id` | 标量 | int64 | 任务枚举，映射表在 configs/data.yaml |
| `episode_id` | 标量 | int64 | 域内枚举；跨域唯一性由上层拼接时加域前缀保证 |
| `frame_id` | (k+1,) | int64 | episode 内帧号，缓存 key 组件 |
| `is_intervene` | 标量 | bool | 人工干预帧标记（数据自带，逐帧） |
| `episode_name` | 标量 | str | episode 目录名（划分哈希、调试追溯用） |

说明：

- 上表为 clip 形态（k>0，Stage 1/2）的设计契约；**当前实现为 k=0
  单帧**（Stage 0），各字段不含时间维（如 `rgb` 为 (3,224,224)、
  `action` 为 (16,)、`frame_id` 为标量），clip 组装层落地时升维；
- `depth/mask/teacher` 逐帧给出（Stage 1/2 要对每个 ẑ 步解码监督）；
- 批次化由 collate 完成，dataset 单样本不含 batch 维；
- 相机路数：当前默认两域均 `camera_front`（1280×720），但两域该机位
  **视角不一致**（real 斜视 / sim 近垂直顶视），重新配对待裁决，
  见 §12.9；real 辅助相机命名跨批次互换，见 §12.1。

## 5. 归一化约定

- RGB：**[已裁决 2026-09-11，二次裁决] 方案 B'**——等比缩放填满宽度
  （1280×720 → 224×126）后上下填黑补 224²（黑边按 patch 对齐：上 3 行 /
  下 4 行 patch），再 ImageNet 归一化（DINOv2 契约）。两域同为 16:9 ⇒
  黑边占比恒定（7/16 行），不是域标签；无畸变、全画幅（real 斜视视角中
  杯架贴在画面边缘，裁剪方案会丢）。直接 resize（扭曲 DINOv2 局部
  几何统计）与剔黑块减 token（破坏 256 token 恒定接口 + 16×9 奇数行
  使 VLA adapter 2×2 pooling 无法整除）均被否；完整讨论记录见
  `src/sawvla/data/image.py` 模块头注释。实现：`LetterboxPreprocessor`。
  **深度监督离线产物必须施加完全相同的填黑几何**（填黑区 mask=0）；
- depth/teacher：disparity 归一化到 [0,1]，per-frame 统计随预处理产物
  落盘（供反归一化与可视化）；
- actions/proprio：归一化统计（均值/方差或 min/max，per-domain 还是全局
  **待裁决**）作为预处理产物落盘；
  dataset 读原始值 + 查表应用，**不在线估计统计量**。

## 6. 帧存储与 clip 在线组装

- 离线预处理只落盘**逐帧记录** + episode 索引（帧号 → shard 内偏移）；
- clip 在**线**组装：给定起点 t 与 frameskip s，取帧
  `t, t+s, …, t+k·s`，动作取对应 k 段；
- frameskip s ∈ {1,2,4,8} 在训练时随机采样（configs/model.yaml
  `transition.frameskip` 的既定语义），因此**禁止**在离线侧把帧对/帧段
  关系烤死——这是本设计对 guideline §4.2-1 措辞的明确化；
- 推论：存储格式必须支持 episode 内任意帧的随机访问（LMDB 或带索引的
  WebDataset）。

## 7. Stage 1 的 latent 缓存对齐

- Stage 0 结束后 E 冻结，一次性前向全部帧，z 落盘（bf16，
  256×384/帧，估算见 guideline §5-Stage 1）；
- 缓存 key = `(episode_id, frame_id)`，与 schema 同名字段严格一致；
  **action/proprio 不入缓存**，Stage 1 训练时从视图 #1/#2 按 key 联查；
- Stage 1 的 "dataloader" = 缓存读取器 + 上述联查 + §6 的 clip 组装，
  采样器（§3）照常生效；
- Stage 2 解冻 E 后缓存失效，z 回到在线计算——缓存只服务 Stage 1。

## 8. 各 stage 消费矩阵

| 消费者 | 主损失数据 | 对抗流 | 形态 | 域均衡 |
|---|---|---|---|---|
| Stage 0 | #1 + #2 | — | 单帧（k=0） | 可选，默认关 |
| Stage 1 | latent 缓存（源自 #1+#2）+ action/proprio 联查 | —（只挂双探针监控，不治） | clip k∈{1,2,4} | 默认开 |
| Stage 2 | #1 + #2 | #3（=全库，Franka 单 matched 任务） | clip | 开 |
| 域探针/评估 | — | — | #3 上拟合与评估 | — |
| 深度探针 | val 集（#1+#2 的 held-out episode） | — | 单帧 | — |

## 9. 划分与防泄漏

- train/val/test **按 episode 划分**，按 (域 × 任务) 分层
  （guideline §4.2-6），杜绝 episode 泄漏——探针实验对泄漏极敏感；
- #3 的 matched 子集继承同一划分（其 val 部分才是探针评估的合法输入）；
- 划分表落盘为预处理产物（episode_id 列表），dataset 按表过滤，
  不在加载时随机划分。

## 10. 模块落位（实现时）

```
src/sawvla/data/
├── dataset.py      # [已实现] 原子 Dataset 基类 + FrankaReal/SimDataset 按域子类
├── stage0_dataloader.py  # [已实现] Stage 0 专用子类：按 puppet 双臂累计
│                         #   运动量贪心抽稀（τ=0.02 rad，首末帧恒保留）
├── image.py        # [已实现] RGB 预处理基类 + 方案 B' 填黑（含讨论记录）
├── supervision.py  # [已实现] 深度监督策略基类 + Raw（调试）+ 在线 64×64
│                   #   disparity（默认，Stage 0 快速验证路径，不落盘）
├── action.py       # [已实现] 动作/本体感预处理基类 + Franka 16 维关节+夹爪
├── samplers.py     # DomainBalancedSampler
├── clips.py        # 在线 clip 组装（起点 + frameskip → 帧索引序列）
├── cache.py        # Stage 1 latent 缓存写入/读取（key 对齐）
└── stats.py        # 归一化统计产物的读取
```

2026-09-11 实现说明：dataset/supervision/action 四个模块已落地并有单测
（tests/test_dataset.py、test_image.py、test_stage0_dataloader.py，
77 项全绿）。Franka introspection
结论已写入 configs/data.yaml：real 300 eps / ~8.8 万帧，sim 319 eps /
~5.7 万帧；act_dim = proprio_dim = **16**（双臂 7DoF 关节 + 双夹爪，
两域同构）；相机两域默认 camera_front（1280×720；两域视角不一致，
见 §12.9）；深度 uint16
毫米，两域共认 65535 哨兵（real 另有洞）。

2026-09-11 Stage 0 冒烟：scripts/train_stage0.py 已跑通（30 步，
outputs/stage0_smoke/）——深度监督为在线 OnlineDisparitySupervision
（不落盘，§4 depth/mask 两字段即其输出；teacher 字段缺省、λ_teacher=0），
train/val 按 episode crc32 哈希确定性划分（§9 正式划分表落盘前的
临时实现），防漂移锚生效（step0 drift=0 起步递增）。

2026-09-11 运动抽稀（用户裁决）：Stage 0 默认经 Stage0MotionThinnedDataset
按 puppet 双臂 14 关节累计 ||Δq||₂ 贪心抽稀（τ=0.02 rad，首末帧恒保留，
`--motion-thresh 0` 关闭）。动机：30Hz 相邻帧 ~99% 重复、real 有 9.6%
零运动帧，全帧送入浪费算力且放大静止段占比。实测保留率：real 66.3%
（88,451→58,619）、sim 77.7%（56,582→43,987）。运动信号不含夹爪
（两域 EE 量纲不同：real bang-bang [0,1] / sim 米制 [0,0.16]）。

一切参数（路径、白名单、划分表、统计文件）来自 `configs/data.yaml`。

## 11. 待解决依赖（阻塞实现，按序）

1. **§4.1 introspection 收尾**：Franka 侧主体结论已落档（相机/动作/深度/
   EE），剩余：`docs/data_schema.md` 正式落档、动作归一化统计产物；
2. **深度监督构造**（框架已定，见 docs/depth_gt_supervision.md）：统一产物契约 +
   实现 #1（raw+mask 基线）/ #2（教师预刷 + 尺度统一）；本文档 §4 中
   `depth/mask/teacher` 三字段分别对应其产物的
   `sensor_disp/sensor_mask/teacher_disp`。

## 12. 已知数据质量问题登记（Franka，2026-09-11 实测）

> 证据：scripts/check_franka_dataset.py 与 outputs/check_franka{,_ep2}/。

### 12.1 相机命名/机位跨批次不一致（real，2026-09-11 全库核实）

- real 分两采集批次：**0520 批 289 集 / 0821 批 11 集**；
- `camera_front` 分辨率全库一致（300/300 为 1280×720）**但机位不同**：
  0520 批 = 近距离斜视（手臂水平左右伸入），0821 批 = 高位宽视角——
  **分辨率一致不能证明机位一致**，视角核实必须渲染目检
  （证据：outputs/check_camera_views/real_front_12eps.png）；
- 辅助相机更乱：camera_left 在 0520 批为 1280×720 斜俯视、在 0821 批为
  640×480 纯侧视（可见背景显示器）——同名不同机位（证据：用户检验
  pair_simfront_realleft_60.png 时发现）；
- sim 侧无批次结构（319 集各自独立 id），camera_front 抽样核实视角
  一致（近垂直顶视）；
- 结论：real 任何相机的"名字 → 机位"映射只在单批次内有效；跨批次
  使用前必须逐批渲染核实。

### 12.2 real 时间戳损坏

real `timestamp` 差分只有 {0, 1} ms（全库一致）——时间戳不可用，跨信号
对齐只能信任 `_align` 对齐序列本身（当前默认策略即如此，无动作需要）。
sim 时间戳正常（33ms ≈ 30Hz），另有 60Hz `_raw` 原始信号可用。

### 12.3 real metadata 全空

real 无语言指令（sim 有：`Hang the cups scattered on the table on the
cup rack`）。补标策略沿用既定裁决：逐字复用 sim 同任务原文（任务级常数
指令），防止文本通道成为域标签。

### 12.4 real arm_align 是 8 维

real 的 arm_align = 7 关节 + 夹爪（第 8 维与 EE 字段逐位相同，已核实），
sim 为纯 7 维。统一处理：臂曲线一律裁前 7 维 + 独立 EE 字段
（FrankaJointGripperPreprocessor），两域无分支。

### 12.5 EE 信号形态差异（方向已确证一致）

master EE 高值=抓握、0=松开，两域同向（逐帧图像核实）。real 为归一化
bang-bang [0,1]（puppet 实测被物体挡住时读数 <1，如左爪握杯外壁
0.18–0.37，为正常物理现象非故障）；sim 为连续 [0, ~0.16]。归一化统计
落盘后桥接。

### 12.6 深度 65535 哨兵

两域深度图都有 65535 远平面/无效哨兵（real camera_right、sim
camera_left/right 实测出现），掩码上限（real 5000 / sim 10000）将其排除；
real 另有洞（0 值），洞集中在深色机械臂本体（材质吸光）。

### 12.7 通道序两域不同

real = rgb，sim = bgr（文件内 `camera_color_channel` 标记）。dataset 逐
文件读取该标记并全库校验一致，解码时按标记处理，不做统一假设。

### 12.8 sim 拼写错误空目录

`124-hang_cup_on_cup_holer`（0 episode），发现逻辑自动忽略（无 hdf5）。

### 12.9 两域默认相机视角系统性不一致（2026-09-11 发现，处理待裁决）

**现象**（证据：scripts/check_camera_views.py →
outputs/check_camera_views/ 下 view_compare.png、rot180_check.png、
real_front_12eps.png、pair_simfront_realleft_60.png）：
real/camera_front 主批次（0520，289 集）是**近距离斜视**（双臂从画面
左右两侧水平伸入）；0821 批 11 集机位不同（高位宽视角，见 §12.1）。
sim/camera_front 是**近垂直顶视**（双臂从画面上方伸入）。
差异是相机 3D 位姿（俯仰角），不是平面内旋转/缩放——2D 变换
（含随机缩放增强）无法对齐，RGB-D 重投影 / 4D GS 等"真修"方案
工程量和副作用均不可接受（讨论记录 2026-09-11）。

**测量方法教训**：跨域跨 episode 的 SIFT/边缘 chamfer 匹配在此
不可用（场景内容不同 + 桌面边缘 180° 近似对称 + 跨域外观差），
曾得出 rot≈1°/scale≈1.28 的伪结论并被推翻；跨域几何差异只能用
语义地标（手臂、杯架位置）人工核实。

**候选修法（均待裁决）**：
1. ~~real/camera_front ↔ sim/camera_top~~（视角结构接近但非同机位）；
2. ~~real/camera_left ↔ sim/camera_front~~（2026-09-11 用户检验 60 组
   对照图后否决：real/camera_left 跨批次机位不一致，见 §12.1）；
3. 接受视角差为内容级域差，按 guideline 先量后治——Stage 0/1 后以
   域探针量化危害再裁决；是否剔除 real 0821 批 11 集（保持 real 域内
   视角单一）一并届时裁决。
