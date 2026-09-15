# 世界模型技术路线 Guideline v2

> 本文档是世界模型研究项目的完整技术路线与工程约束说明，供 coding agent / 研究工程开发直接使用。
> **本文档为当前唯一有效版本**（v1 已废弃，勿参考）。
> 所有标注 `[待本地核实]` 的条目，指需要在本机数据集上 introspect 后确认的 schema 细节，见 §4.1。

---

## 0. 项目概述

### 0.1 目标

训练一个机器人桌面操作世界模型：

1. **编码**：将当前帧 RGB 编码到连续隐空间（patch 级 token，非全局向量）；
2. **递推**：在隐空间内，以时序动作条件注入的转移模型预测下一帧隐空间表达；
3. **几何可读**：隐空间向量可解码出（下一帧的）稠密深度图；
4. **服务 VLA**：隐空间表达后续用于指导 VLA（Vision-Language-Action）策略；
5. **域隔离**：以"深度"为隔离层，使隐空间不携带"RGB 来自仿真域还是真实域"的信息；辅以 register 分区（域信息给指定槽位居住，§2.4）与针对性 GRL（只清域无关槽位，Stage 2），并以三通道域探针量化披露残余泄漏。动机：任务执行与 RGB 画质无关，模型应关注物体几何本身；避免 photo-realistic 图像生成的代价及生成不完美对特征提取的负面影响。

### 0.2 数据

- **真实域**：RoboMIND2.0-Franka-Part-1（双臂 Franka 工作站，真实环境采集，HDF5 统一格式，深度来自 RealSense D435if，**存在空洞与噪声**，洞集中在深色机械臂本体）。
- **仿真域**：RoboMIND2.0-Franka-sim（同款双臂 Franka 在仿真环境完成同一任务；深度完美稠密，带全套相机内外参）。
- **2026-09-11 数据集切换（用户裁决）**：默认数据集从 Tienkung（天工人形）切换为 Franka——同机器人、同 matched 任务（hang_cup_on_cup_holder）、EE 同构的"完美对应"数据，必须在完美对应的数据上做才有意义；Tienkung 归档，后续有空再议。
- **采样率（2026-09-15 用户核实裁决）**：real 名义 **15fps**（66.6ms；时间戳已损坏，名义值是唯一可信口径，见 dataloader.md §12.2）；sim **30fps** ⇒ 训练帧流统一 15fps，sim 侧直接抽帧（每 2 帧取 1，不在位置编码上做区分）；自建仿真环境（src/simulation）控制频率同步设为 15Hz（simulation.yaml `decimation: 4` @ 60Hz 物理步，Δt=66.7ms）。
- 两份数据集**已下载到本机**，路径通过环境变量 `ROBOMIND_ROOT` 定位，禁止联网重复下载。

### 0.3 核心难点

真实域深度监督信号不完美：有真缺失（无返回值）、有噪声/飞点，且空洞分布与物体材质强相关（透明、反光、黑色吸光、边缘——恰是操作任务中最需要几何的对象）。需要在此条件下完成训练，且不让 sim/real 深度质量分布差异把域信息通过监督梯度泄漏进隐空间。

---

## 1. 资源与约束（硬约束）

| 项目 | 约束 |
|---|---|
| OS | Ubuntu 24.04 LTS |
| GPU | 单卡 RTX 4070 Ti Super，**16GB 显存**，Ada Lovelace（支持 bf16 / FlashAttention-2） |
| VL 基模 | `Qwen/Qwen3-VL-2B-Instruct`（视觉塔 SigLIP2-Large ≈300M，2×2 merger 压缩视觉 token，DeepStack，动态分辨率；精确配置以本地 `config.json` 为准） |
| 数据集 | RoboMIND2.0-Franka-Part-1 + RoboMIND2.0-Franka-sim，本地已下载 |
| 精度策略 | 全链路 bf16 混合精度；优化器状态必要时 8-bit |

16GB 显存的推论（违反将导致 OOM，务必遵守）：

- Qwen3-VL-2B **禁止全量微调**（bf16 权重 ≈4–5GB，AdamW 的 fp32 m+v 状态 ≈16GB，不可行）。只能用 **LoRA + 8-bit 优化器 + 冻结视觉塔 + 梯度检查点**。
- 世界模型主干为 DINOv2-S 量级；教师模型离线推理选 Depth Anything V2-L（335M，可放下）。
- 所有训练脚本默认开启 `torch.utils.checkpoint`、梯度累积；batch size 通过累积等效放大。

---

## 2. 总体架构

```
                ┌────────────────────────────────────────────┐
 RGB_t ───────► │ Visual Encoder E (DINOv2-S/14-registers)   │ ──► z_t patch (16×16×384，局部通道)
 RGB_{t+1} ───► │  （同一 Encoder，推理时只跑当前帧）           │ ──► r_t registers (4×384，全局槽位，§2.4)
                └────────────────────────────────────────────┘
                                  │
        action_t ──► Action Adapter (MLP→action tokens) ──► ┌───────────────┐
                                  └────────────────────────►│ Transition T   │──► ẑ_{t+1}
                                                            │ (Transformer)  │
                                                            └───────────────┘
                                  │                                    │
                ┌─────────────────▼──────────────┐   ┌────────────────▼─────────────┐
                │ Depth Decoder D（共享，仅接 z） │   │ Domain Discriminator（GRL）   │
                │ 输入 z_t 和 ẑ_{t+1}，输出 64×64│   │ 只挂 reg_agnostic 通道（§2.4）│
                │ 稠密 (μ, σ)                    │   │ Stage 2 上线，梯度反转进 E    │
                └────────────────────────────────┘   └───────────────────────────────┘
                                  │
                VLA 阶段：z_t / ẑ 经 Adapter 注入 Qwen3-VL-2B（§8）
```

> 记号约定：下文 `z` 一律指 **patch 通道**（局部、2D 网格）；`r` 指 **register 通道**
> （全局槽位）。T 递推 register 通道已裁决（2026-09-15：每步同时预测 ẑ 与 r̂，
> 见 §6-Stage 1）；WAM 对 VLA 的输出接口仍是恒定的 256 个 patch token，
> register 通道是否进 VLA 留 Stage 3 裁决。

### 2.1 隐空间规格与 uv 绑定（本项目的量化权衡结论）

**隐空间大小**：每帧 256 token × 384 维 = **98,304 维**（bf16 ≈192KB/帧）。参照系：

| | 本设计 | DINO-WM | V-JEPA 2 |
|---|---|---|---|
| 输入分辨率 | 224×224 | 196×196 | 256×256 |
| 空间网格 | 16×16 | 14×14 | 16×16 |
| token 数 × 维度 | 256×384 ≈ **9.8 万维/帧** | 196×384 ≈ 7.5 万维/帧 | 256×1408 ≈ 36 万维/帧 |
| 编码器 | DINOv2-S/14，**微调** | DINOv2，冻结 | ViT-g/16（1B），冻结 |
| 位置编码 | 2D sincos（可插值），T 内 2D-RoPE | 绝对位置编码 | 3D-RoPE（时间/高/宽三轴） |
| 预测器 | ≤100M，从零 | 19M（6 层 ViT），从零 | 300M（24 层/d1024/block-causal），从零 |

**权衡原则（已实现为下文的硬规格）**：

1. **空间网格是深度可解码性的下限，不可压缩**：网格低于 ~12×12 时，夹爪、线缆、物体轮廓级别的几何在原理上不可恢复。要省参数/算力，只压通道 d（384→256 线性压缩可行），不动网格。
2. **稠密深度可解码的需求排除了极致压缩路线**：bisimulation 式压缩（196×32）或 object-centric 槽位（6×128）能跑规划，但无法还原稠密深度图，与本项目目标冲突，不采用。
3. **动力学侧的成本用训练技巧控制，而不是砍网格**：frameskip（帧跳，避免相邻帧平凡的复制解）+ 动态区域损失加权 + block-causal 注意力。
4. 显存富余时的升级顺序：先升通道 d（384→768），再升输入分辨率，不动网格下限。

**uv（图像横纵向）绑定**：有，且必须保持。实现方式：

- 每个 patch token 通过 2D 位置编码绑定到固定的 14×14 像素块，2D 网格结构全程保留，**禁止全局池化/全局单向量**（DINO-WM 消融：全局向量化显著掉点）。
- 绑定的常规载体是位置编码而非坐标通道；如需更强绑定：解码器入口可拼接归一化 uv 坐标通道（帮助深度边缘对齐）；多相机 setup 必须加 camera embedding token。
- 动作/本体感是全局量，只加时间维位置编码（V-JEPA 2-AC 做法），不携带空间编码，通过注意力/AdaLN 调制 patch token。

**patch 网格不是隐空间的全部**：E 的输出还有 4 个 register token（全局槽位通道），
其角色划分与消费纪律见 §2.4——局部信息走 patch、全局信息走 register，两通道分离。

### 2.2 关键设计决策（已定，含理由）

| 决策 | 选择 | 理由 |
|---|---|---|
| 隐空间形式 | 连续 latent，patch 级 token，规格见 §2.1 | 连续 latent 配 GRL 梯度反转顺滑、对 VLA 接口友好；patch 表征保留操作任务必需的空间细节 |
| 编码器初始化 | **DINOv2-S/14（registers 版）初始化，全程可微调，禁止冻结** | (i) DINOv2 patch 特征稠密几何可探针性久经验证，是 DINO-WM 的世界模型底座；(ii) 深度教师 Depth Anything V2 的主干即 DINOv2——教师与 E 特征同族，教师深度对该特征空间高度可解码，Stage 0 收敛快；(iii) 与 DINO-WM 的关键差异：GRL 必须能反传进 E 除域、Stage 0 深度损失必须塑形 E，冻结后两者无处着力 |
| **不使用 VLM 参数初始化** | 不以 CLIP/SigLIP 类语言对齐模型作为 E 的默认初始化（仅列为消融） | 语义对齐压缩丢弃的正是世界模型需要的像素级/几何信息；JEPA 系（V-JEPA 2 从零视频 SSL）、DINO-WM（DINOv2）均不用 VLM 参数；VLM 初始化是 VLA 策略（RT-2/OpenVLA）的惯例，不是世界模型的惯例；且与 Qwen3-VL 的语义分工重复 |
| 动作注入 | 动作**只进转移模型 T，不进编码器 E** | 保证 z 是纯观测表征，便于静态重建与探针诊断 |
| 深度解码器输入 | **只允许 z，禁止 encoder 中间层 skip connection** | 若带 skip，深度走 RGB 捷径，z 的几何含量探针失效，"几何在隐空间"的整个论证崩塌 |
| 深度输出/监督分辨率 | **64×64（上限 112×112），不做满分辨率** | z 的空间信息上限是 16×16 网格，超过 ~4–7 倍上采样的内容纯属解码器幻觉，强监督满分辨率只会污染梯度并被 σ 吸收成噪声；DINO-WM 的 16× 上采样解码器也只用于可视化、不回传。满分辨率仅用于评估与可视化 |
| 深度解码器共享与梯度 | 当前帧与预测帧共享同一个 D；动态阶段解码损失对 T/E **stop-gradient**（只训练 D 本身） | DINO-WM 消融：解码损失反传进预测器会显著伤害下游性能（PushT 0.80 vs 0.92）；静态阶段例外，见 §6 |
| 域对抗位置 | **GRL 只挂 reg_agnostic 通道**（register 分区，§2.4），patch 与其它槽位只探针监控、不对抗 | 硬去除纠缠的几何/机位信息必然误伤任务性能（过杀）；隔离优于摧毁——域信息给指定槽位（reg_domain）居住，下游按通道自选。T 已裁决递推 register（2026-09-15）⇒ ẑ 侧 reg_agnostic 通道同样挂 GRL |
| register 分区 | 4 个 register 分角色：`[0,1]`=域信息槽位（被动锚定携带，不主动预测域），`[2,3]`=域无关槽位（全局信号头唯一读取处）；索引写在 model.yaml `encoder.register_roles` | 全局/局部分离：patch 已由深度损失显式塑形局部信息，register 天然是 12 层注意力的全局汇聚通道；分区让域信息有指定住所、可量化监控（探针读数=泄漏量），而不是满隐空间流窜 |
| 深度教师 | Depth Anything V2-L（相对深度），离线对**两个域同一模型**推理，缓存稠密伪深度 | 同一教师 ⇒ 监督分布天然同域化；稠密 ⇒ 消除掩码不对称；相对深度 ⇒ 天然域不敏感 |
| 变分正则 | 可选小权重 KL（β-VAE 式），默认关闭 | 先跑通确定性版本，出现后验塌缩/外推不稳再加 |

### 2.3 模块规格（默认值，可调）

> T 与 D 的逐层结构、参数-数据匹配论证与算力时间预算，见《隐空间与监督设计.md》§7/§8/§9。

- **E**：DINOv2-S/14（registers 版，d=384，~22M）初始化，224×224 输入 → 16×16=256 patch token；**初始化后全程可微调，禁止冻结**（理由见 §2.2）。完整序列 = 1 CLS + 4 register + 256 patch（261 token），出口三通道（cls / registers / patch，`forward_tokens`），消费纪律见 §2.4。Stage 0 用低 lr（≤1e-4）并加**防漂移锚**：保留一份冻结 DINOv2 副本，对 E 的 **patch + 全部 register** 输出加小权重特征蒸馏正则（CLS 无人消费、不锚；register 纳入锚定是 2026-09-14 裁决——锚只防漂移不除域，教师对两域输入的编码差异会被一致地保留为"教师格式"），防止深度监督把 SSL 通用先验冲掉；同时监控 SSL 通用性探针与深度探针双指标。显存富余可升 ViT-B/14。消融项：SigLIP2（Qwen3-VL 视觉塔）初始化、MAE/VC-1 初始化、随机初始化。
- **T**：默认 8 层 Transformer、d=384（~20M 参数，上限 ≤100M；参照：DINO-WM predictor 19M/6 层，V-JEPA 2-AC predictor 300M/24 层/d1024/block-causal），**一律随机初始化从零训练**——没有现成权重匹配本项目动作空间，所有路线（含 V-JEPA 2-AC）均如此。输入 = z_t 的 256 个 patch token + 1 个本体感 token + k 个逐步动作 token（**每步动作向量独立 MLP 升为 1 个 token，k 步 = k 个 token，禁止整块压缩**——逐步 token 的因果语义干净，第 j 步预测只能 attend a_{≤j}，详见 `overall_tensor_flow.md` §1.2/§3.2；动作维数 `[已核实 2026-09-11，Franka]`：双臂 7DoF 关节位置 + 双夹爪 = **16 维**（action=master 指令、proprio=puppet 实测，两域同构；夹爪已裁决纳入，方向两域一致：高=抓握；详见 configs/data.yaml curves 段与 dataloader.md §12.4/12.5）。uv 绑定按 §2.1：patch token 加 2D 位置编码，动作/本体感 token 只加时间维位置编码；条件方式默认 token 拼接 + AdaLN 二选一，做消融。**frameskip**：数据处理引入帧跳参数（DINO-WM 做法），避免相邻帧过于相似导致平凡复制解；训练时可选**动态区域损失加权**（按相邻帧 latent 差加权），防止容量浪费在静止背景。
  > **[已裁决 2026-09-11]** 动作 token 化采用逐步 token（每步 1 个，不整块压缩）；T 的因果语义按 `overall_tensor_flow.md` §3.2（block-causal，禁止看未来动作），因果泄漏单测（§3.4）必须实现。模块边界原则：**WAM 模块（E/T/D）不感知 VLA 内部实现**；是否压缩、如何压缩是 VLA 侧 adapter 的内部事务，WAM 对外接口恒定输出 256 个 patch token。
- **D**：轻量 DPT 式上采样头，双头输出 `(μ, log σ)`。输入只允许 z（§2.2）；输出 64×64（上限 112×112）；可选在入口拼接归一化 uv 坐标通道。
- **Domain Discriminator**：reg_agnostic 通道（2×384 全局槽位，§2.4）→ 3 层 MLP，二分类（real/sim）。容量刻意做小（~1M），防止判别器过强导致 GRL 训练不稳。**patch 通道不做对抗**（隔离优于摧毁，§2.4），只挂探针监控。

### 2.4 模块化边界：register 分区、可插拔信号与探针（2026-09-14 裁决）

本节回答"模块化思路里**能插拔的是什么、不能替换的是什么**"。

**三通道输出**（`DINOv2Encoder.forward_tokens`，单次前向）：

| 通道 | 形状 | 角色 | 消费者 |
|---|---|---|---|
| `patch` | 256×384 | **局部通道**：2D 空间网格，深度可解码性的载体 | D（唯一输入）、T、VLA adapter |
| `registers` | 4×384 | **全局槽位**：12 层注意力的天然全局汇聚通道 | 全局信号头、域探针、Stage 2 GRL |
| `cls` | 384 | 闲置（无人消费、不锚定） | — |

**register 分角色**（初始化对称，角色纯靠约定 + 损失塑形，索引可配）：

- `[0,1]` **域信息槽位**：被动锚定携带域信息（教师对两域输入自然给出不同编码），**不主动加域分类损失**——先量后治，探针读数不够再升级；
- `[2,3]` **域无关槽位**：**全局信号头的唯一读取处**；Stage 2 GRL 只挂这里。本体感、任务量等帧级全局信息的规定住所。

**可插拔：监督信号注册表**（`src/sawvla/signals/`）。全局信号是插拔件，新增信号三步、trainer 零改动：① 写 `SignalHead` 子类（`reads` 指定读哪个通道 / `forward` / `target(batch)` / `loss`）；② `@register_signal("名字")`；③ model.yaml `signals` 节加一行（enabled / weight）。示例：做举杯任务就注册"杯到桌面高度"预测头，做自身状态监督就注册关节回归头。已注册首信号 **joint_pos**：双臂关节位置回归 14 维（**剔除夹爪**——sim 夹爪是连续行程、real EE 是二值开合，跨域不同纲；臂关节同为绝对关节角、跨域同纲，天然弱域对齐锚），读 reg_agnostic，weight 0.1。**预测端对等（2026-09-15 裁决）**：Stage 1 起信号同时挂 T 的预测输出（r̂/ẑ，与 E 侧同一批头同一权重），预测侧梯度进 T（选项 A，`stage1.signals_on_prediction`）；唯一例外是 T 不预测的 CLS——预测侧 ctx 的 CLS 置零，未来注册读 CLS 的信号需先裁决预测侧语义。

**独立组件：域探针**（`DomainProbe`，刻意**不走**信号注册表：它是仪器不是损失，特征 detach、独立小优化器，梯度绝不进 E）。探针体系三档（2026-09-15 扩展），分工是三个不同的问题：

- **mean-pool 线性**（三通道 patch / reg_domain / reg_agnostic，TB `probe/acc_*`）：域信息是否**平凡可读**（总量）。reg_domain 应高（设计上就是域信息的住所）；reg_agnostic 目标 ≈ 50%（Stage 2 GRL 的工作对象）；patch 是泄漏监控。
- **逐 token 热图**（patch 通道 256 位置各一个独立线性头，不注意力路由；TB `probe/tok_acc_{mean,max}` 标量 + 每个 val 周期 val 集累计的 16×16 热图 `probe/patch_leak_map`）：域信息漏在**哪里**——区分画风泄漏（背景/光照区，良性）与内容泄漏（杯子/机械臂，危险）；已知局限：逐 token 各 55% 聚合后仍可 100%（冗余累积，不替代汇总读数），且测不到关系型泄漏（机位差异在 token 间相对配置里，由 register 通道兜住）。
- 残差关系：attention 审计探针（上限档）暂未实现，需要时再加。

设计动机（用户原话）："很难做到完全不含信息，但可以做到尽量少含信息，并且知道自己大概泄露了多少域信息，对下游用户的信心也有显著提升。"探针与 50% 的差值 = 该通道域信息含量的运营指标，全程记录进实验台账。

**不可替换件（硬规格，改动即破坏项目论证）**：

- patch 通道规格 256×384 与 uv 绑定（§2.1）——WAM 对外恒定接口；
- E 全程可微调；D 只读 patch（禁 skip connection）；动作/本体感 16 维契约；
- 分区纪律：全局信号只读 reg_agnostic（读 patch 会把局部几何泄进全局通道，读 reg_domain 会被域污染）；
- 深度语义：z-depth、毫米、1mm 整数量化、无效=0（两域同纲，见 src/simulation/README.md 数据语义约定）。

**可插拔件（换实现不动框架）**：全局信号头（注册表）、域探针、数据域子类（dataloader 按域各携规则）、仿真侧机器人（robots.py 注册表）与相机机位（simulation.yaml）、RL 后端（rl/facade.py 包装层，当前 rsl_rl）。

---

## 3. 术语与符号

- `d`：传感器深度（real 有洞有噪；sim 完美）。`M`：有效性掩码（1=有值）。
- `t`：教师伪深度（两域均稠密）。`μ, σ`：解码器输出的预测深度与不确定性。
- `z_t`：编码器 **patch 通道**输出（256×384，局部）；`r_t`：**register 通道**（4×384 全局槽位，角色见 §2.4）；`ẑ_{t+1}`：转移模型预测；`sg(·)`：stop-gradient。
- 深度统一转为 **disparity（逆深度）** 空间计算损失（近处分辨率高，且与 scale-invariant 损失兼容）；存储用 uint16 量化。

---

## 4. 数据管线

### 4.1 第 0 号工程任务：数据集 introspection（先于一切）

数据集 README 无有效信息，**必须先对本地文件做 schema 探查并落档**：

1. 用 `h5py` 遍历若干 episode 文件，打印完整 group/dataset 树、dtype、shape；
2. 确认并记录：相机路数与名称（头部/腕部？）、RGB 分辨率与 fps、深度 dtype 与量纲（消费级 RGB-D 通常为 uint16 毫米）、深度与 RGB 是否已对齐（registration）、动作/本体感的键名与维度、episode 长度分布、任务标注字段；
3. 抽样可视化：RGB、深度、深度有效性掩码、洞的空间分布（确认洞与物体材质的相关性，这决定 §5 的必要性判断）；
4. 产出 `docs/data_schema.md` 与 `configs/data.yaml`，后续所有 dataloader 以此为唯一事实来源。

> **进度（2026-09-11，Franka 切换后）**：主体已完成并落入 configs/data.yaml——相机两域统一 camera_front（1280×720 16:9；**后发现两域该机位视角不一致：real 主批次斜视 / sim 近垂直顶视，处理待裁决**，见 dataloader.md §12.9），深度 uint16 毫米（real 有洞、65535 哨兵；sim 稠密、同哨兵），动作/本体感 = master/puppet 双臂 7DoF 关节 + 双夹爪（**16 维**，两域同构，EE 已裁决纳入），episode/任务数与帧量已核实（real 300 eps / 8.8 万帧，sim 319 eps / 5.7 万帧，§9.1）。**未完成**：registration 状态（sim 有内外参、real 无）、洞-材质相关性可视化（第 3 条，Franka 侧初查：洞集中在深色机械臂本体）、`docs/data_schema.md` 正式落档。

### 4.2 离线预处理（一次性，产出缓存）

按顺序执行，全部产物落盘为 WebDataset/LMDB shard（避免训练时重复解码 HDF5）：

1. **抽帧与对齐**：RGB + 深度 + 动作 + 本体感按 `_align` 对齐序列同索引对齐（Franka real 时间戳已损坏，见 dataloader.md §12.2）；RGB 按方案 B' 填黑补 224×224（dataloader.md §5，**深度/掩码离线产物施加相同填黑几何**，填黑区 mask=0）；按 frameskip 参数采样帧对/帧段。
2. **有效性掩码**：`M = isfinite(d) & (d > d_min) & (d < d_max)`，`d_min/d_max` 取传感器量程（introspection 确认）。**只此一条规则，禁止额外手工规则**（§附）。
3. **时序中值补背景**（近乎零成本的白送增益）：桌面场景大面积静止，对同一相机位姿的静止片段做逐像素跨帧中值，得到干净背景深度，用于填充静态区空洞；动态区（机械臂、被抓物体）不补，交给教师监督。
4. **教师伪深度**：Depth Anything V2-L 离线推理全部帧（两域同一 checkpoint），输出相对深度，per-frame 存 uint16 量化 disparity；batch 8 @ 518px，16GB 可放下。
5. **监督目标降采样**：传感器深度、教师深度、掩码一律**掩码感知降采样**到 64×64（无效像素不参与池化，池化后按有效比例重建掩码），与解码器输出分辨率一致；满分辨率原图保留仅用于评估与可视化。
6. **域标签与划分**：**按 episode 划分** train/val/test，按 (域 × 任务) 分层，杜绝 episode 泄漏（探针实验对泄漏极敏感）。

### 4.3 在线加载

- RGB 归一化用 ImageNet mean/std；disparity 归一化到 [0,1] 并记录 per-frame 统计（供反归一化与可视化）。
- 每个样本返回：`rgb_t, rgb_{t+1}, d_t(64×64), M_t(64×64), teacher_t, teacher_{t+1}, action_t, proprio_t, domain_label, episode_id`。

---

## 5. 深度监督方案（本项目核心，务必按此实现）

设计原理：把"不完美的深度"按缺陷类型拆三层，分别处理。**"哪些像素有梯度"的差异本身就是域信号**，因此目标是让两域的监督在分布上尽量同构。

### 5.1 三层处理

| 缺陷类型 | 处理 | 说明 |
|---|---|---|
| (a) 真缺失（NaN/0/超量程） | **硬掩码**，零梯度 | 没有监督目标的地方本就不该有梯度；这是唯一允许的掩码规则 |
| (b) 有值但不可靠（飞点/反光伪值/远距噪声） | **异方差不确定性软降权**，学习式 | 解码器输出 σ，用 Laplacian NLL；传感器失效模式高度规律，网络自己学降权，σ 图同时是诊断工具。**禁止手工阈值规则** |
| (c) 空洞区监督缺失 | **教师稠密蒸馏补监督** | 空洞与物体材质强相关（透明/反光/黑色/边缘），纯掩码 = 系统性不学关键物体的几何，必须用伪标签补上 |

### 5.2 损失函数

```
L_depth = λ1 · L_metric + λ2 · L_teacher + λ3 · L_grad + λ4 · L_smooth

L_metric  = mean_{M=1}( |d − μ| / σ + log σ )                # 异方差 Laplacian NLL，仅有效点
L_teacher = mean_all( |align(t) − μ| )                       # 全图稠密；align = per-image 最小二乘拟合的
                                                             # scale-and-shift 对齐（MiDaS 式）
L_grad    = Σ_s mean_{M_s=1}( |∇μ_s − ∇d_s| )                # 多尺度梯度损失（s = 4 个下采样层级，
                                                             # 掩码同步降采样）——"损失侧低通"，
                                                             # 替代对目标做低通滤波
L_smooth  = mean_{M=0}( |∇μ| · exp(−|∇rgb|) )                # RGB 边缘感知平滑，仅空洞区
```

默认权重：`λ1=1.0, λ2=1.0, λ3=0.5, λ4=0.1`（写进 config，可调）。所有损失在解码器输出分辨率（64×64）上计算。

要点：

- 教师深度是**相对深度**，`align` 步骤不可省；`L_metric` 提供度量锚定。第一阶段可先只用 `L_teacher + L_grad + L_smooth` 跑通，再接入 `L_metric`。
- 传感器深度在监督中的角色被刻意降级为"可选度量锚"，因为它的噪声/掩码分布是域标签；教师同分布输出才是主监督。
- `L_smooth` 的 RGB 引导项取自原始 RGB（降采样到 64×64），这是唯一允许 RGB 进入深度监督计算的位置——注意它不经过 E，不产生域泄漏。
- 全程 disparity 空间计算。

---

## 6. 分阶段训练流程

每阶段有明确**验收门（gate）**，不达标不进入下一阶段。所有 gate 指标写入 config，结果登记到 `docs/experiments.md`。

### Stage 0：静态几何重建（当前帧 z_t → 当前帧深度）

- 训练 E + D（σ 头在内），损失 = `L_depth`。**此阶段深度损失正常反传进 E**（它是 E 的唯一塑形信号）。
- **训练脚本 [已实现 2026-09-11；2026-09-14 扩展]**：`scripts/train_stage0.py`——ConcatDataset(FrankaReal+FrankaSim) 单帧，在线 disparity 监督（OnlineDisparitySupervision，λ_teacher=0 mask-only 基线，教师产物未生成前的临时路径，裁决记录见 depth_gt_supervision.md §3），防漂移锚（冻结副本 + 0.1 蒸馏，**覆盖 patch+register 全槽位**），bf16，episode 级 crc32 确定性 train/val 划分（划分表落盘前的临时实现），周期 val + 可视化 + checkpoint。**2026-09-14 扩展**：全局信号注册表上线（首信号 joint_pos 14 维，读 reg_agnostic，§2.4）；三通道域探针 DomainProbe 常驻监控（只量不治，§2.4/§7.3）。冒烟已跑通（outputs/stage0_smoke/）。
- 默认超参：AdamW，lr 1e-4（DINOv2 预训练初始化，低于从零训练的 3e-4 量级），cosine，wd 0.05，bf16，batch 32（累积等效 64），224px。
- 防漂移锚生效中（§2.3-E），蒸馏正则小权重起步。
- 显存估算：DINOv2-S + DPT 头 ≈ 6–8GB，安全。
- **验收门**（在高置信有效像素上评估，评估掩码规则可与训练不同，仅用于指标）：
  - real 域：AbsRel、δ<1.25 达到基线水平（先跑一个 mask-only 基线作对照；本方案 AbsRel 应相对基线显著下降，目标 ≥10% 相对改善，`[按实测调整]`）；
  - **深度探针**（冻结 E，线性/浅层探针）指标达标——这是"隐空间含几何"的客观证据；
  - SSL 通用性探针（冻结 DINOv2 副本特征作对照）无明显退化——防漂移检查；
  - 可视化检查：透明/反光物体区域不再放飞（对比 mask-only 基线）。

### Stage 1：动作条件转移模型（z_t + a_t → ẑ_{t+1}）

**2026-09-15 裁决定稿**：

- **上下文 = 2 帧**（滑窗；1 帧对物体速度不可观测，2 帧是速度可观测下限；更长上下文留消融）；clip 采自运动抽稀帧流、**不跨 episode**。
- **T 递推 register 通道**：每步同时预测 ẑ（patch）与 r̂（register，全局槽位随时间演化）；block = 256 patch + 4 register + 1 本体感 + 1 动作。
- **自由展开（feed-back）**：上下文块带真值 z/r/proprio，展开块回喂 ẑ/r̂、**只带动作 token**（未来本体感不可知 ⇒ 可学习 null 向量填充）；固定最大展开 K=4 步、逐步监督（等效覆盖 unroll k∈{1,2,4}）。
- **预测端监督对等（选项 A）**："T 预测输出应与 E 编码输出尽量不可分辨"——joint_pos 及未来注册的全局信号同样挂 r̂/ẑ 且**梯度进 T**（塑形预测；`stage1.signals_on_prediction.grad_into_t: false` 即选项 C 消融）；深度 D 同样解码 ẑ 但**对 T 保持 stop-grad**（只训 D + 监控；DINO-WM 消融惯例，§2.2——讨论结论：L_dyn 是"逐维相同"的充分条件，深度梯度给 T 提供的是"投影相同"的捷径，风险收益不成比例；信号头小、目标低维，直接进 T 是便宜保险）。
- **时间口径统一 15fps**：real 名义 15fps（§0.2）；sim 30fps 抽帧到 15fps（先抽帧后运动抽稀，τ 语义同速率对齐）；帧内时间注入 = 帧索引 embedding（真机时间戳不可信）。

**实现 [2026-09-15，已落地并冒烟通过]**：`scripts/build_latent_cache.py`（E 冻结前向落盘三通道 latent + rgb + 64×64 disparity 目标 + 动作/本体感，每 episode 一文件 + index.json，划分与 Stage 0 同 crc32 规则）→ `src/sawvla/data/latent_cache.py::LatentClipDataset`（滑窗 C+K=6 帧，LRU 加载）→ `scripts/train_stage1.py`（自由展开 K=4；L_dyn patch+register；信号真值侧训头/预测侧进 T；D 解码 z 与 ẑ 均 stop-grad 只训 D；**域探针挂预测输出**——E 冻结后 z 侧读数是常数，ẑ 的泄漏量才是监控对象；`--overfit-batch` 单 batch sanity 已通过）。T 扩展：`transition.py` 新增 `forward_with_registers / unroll_with_registers`（n_reg=4，register 头零初始化恒等起步，因果泄漏单测钉死）。pytest 101 全绿。

**原始设计条目（仍为默认超参依据）**：

- 冻结 E（或 0.1× lr），训练 T。动力学损失：
  ```
  L_dyn = smooth_l1( ẑ_{t+1}, sg(z_{t+1}) ) + 多步展开版（目标一律 sg；register 通道同构一份 r̂ 损失）
  ```
  frameskip 与动态区域加权生效（§2.3-T；运动抽稀已承担 frameskip 的主要角色，clip_stride>1 为等价补充）。
- 共享 D 同时解码 z_t 与 ẑ_{t+1}，**解码损失对 E、T 一律 stop-gradient**（只更新 D；理由见 §2.2；2026-09-15 选项 A 再次确认）。
- **隐空间离线缓存**：E 冻结后，先把全部帧的 z 一次性前向缓存落盘（`[已核实 2026-09-11，Franka]` 两域合计约 14.5 万帧 × 256×384×bf16 ≈ **28GB** 磁盘，加 register/rgb/深度目标约 60GB 以内，全量缓存无压力），T 训练直接读缓存 latent——编码开销从每步摊销变为一次性，Stage 1 提速约 3 倍。
- lr 1e-4，6 帧片段（2 上下文 + 4 展开）batch 8–16，梯度检查点。显存 ≈ 8–12GB。
- **验收门**：1/4 步 latent 预测误差曲线平滑无发散（patch 与 register 分别报告）；预测帧深度（D(ẑ)）在 val 上可视化合理；多步展开无塌缩（z 范数稳定，`z_norm_ratio` ≈ 1）；预测侧探针读数登记进台账。

### Stage 2：联合微调 + 域对抗（消灭残余域信息）

- 全部模块小 lr（3e-5 ~ 1e-4）联合微调；GRL 判别器上线，λ 从 0 在 ~5k step 内 ramp 到 0.05–0.1（**禁止一步加满**）。
- 同时监控两条探针曲线（§7.3），这是判断拔河走向的唯一手段。
- **验收门**：reg_agnostic 通道域探针准确率降至接近随机（50%±5）——**GRL 只清理这个通道**（§2.4）；patch / reg_domain 通道的探针读数量化登记进台账（不设硬门，patch 允许残留）；且深度探针指标不退化超过 5%（退化即回调 λ）。

### Stage 3：冻结世界模型，接入 VLA（Qwen3-VL-2B）

- 见 §8。E、T 全部冻结，只训 adapter + LoRA。

### 全程不变量

- 深度重建损失在任何阶段都不撤（Stage 1/2 中它以 stop-grad 读出头形式存在，持续提供监控与 D 的训练）。
- 每阶段先跑 **overfit-one-batch** sanity（单 batch 能过拟合到接近 0）再全量训练。

---

## 7. 域隔离与对抗策略

目标边界（2026-09-10 与项目负责人确认）：要隔离的是画风——渲染风格、光照、纹理、传感器噪声等低层视觉线索。任务内容、物体类别、场景布局在 z 中可判别是允许的且必需的（z 要服务下游任务），不算泄漏。本节所有"域信息"特指画风信息；判别器与探针实验均按此口径设计（§7.3）。不做"两域完全不可判别"的严格域适应，也不为对齐画风去做 photo-realistic 仿真（仿真只需几何正确 + 动力学合理）。

### 7.1 泄漏机制（实现者必须理解）

泄漏**不依赖模型在推理时看到深度**。深度只以监督形式出现，但只要深度损失的梯度会回传进 E（Stage 0 必须如此，否则深度目标无法塑造隐空间），而监督在两个域上不对称，"编码域信息"就是一条降低损失的可行路径，SGD 会找到它。三条具体通道：

1. **掩码不对称**：real 空洞区零梯度、sim 处处有梯度，而空洞与物体材质强相关 ⇒ E 被推向"两域学不同几何"；
2. **目标有偏**：飞点/多径伪值使 E[监督值|RGB] ≠ 真实深度，且偏差模式是 real 特有 ⇒ 最优预测函数域相关；
3. **σ 校准**：σ 的最优值取决于噪声水平，噪声水平本身是域标签 ⇒ E 有激励保留域信息帮助 σ 校准。

推论：推理时无深度输入 ≠ 无泄漏；泄漏的充要条件是"损失函数对两个域不对称 + RGB 含域线索"。

### 7.2 对策组合（按优先级）

1. **教师稠密化（主手段，已采用）**：两域同一教师 ⇒ 监督稠密（消灭掩码不对称）且同分布（消灭噪声统计不对称）；scale-and-shift invariant 损失与多尺度梯度损失进一步让最优预测函数域无关。**这是"弄脏 sim 深度"的替代方案，工程量仅为一次离线推理。**
2. **损失侧稳健化（已采用）**：σ 加权吸收残余噪声不对称；多尺度梯度损失强调两域一致的低频几何。
3. **GRL 域对抗（扫尾，不是替代）**：只负责清理损失对齐后的残余泄漏，以及动力学损失本身的域不对称（sim/real 物理与成像动态不同，此通道与深度无关，只能靠 GRL）。

**为什么不允许"只靠 GRL"**：深度损失奖励域相关特征、GRL 惩罚域相关特征，两者拔河。已知失效模式：(i) 欠收敛残余可判；(ii) 过杀——监督不对称时强制不变性会牺牲任务性能（典型场景：透明物体几何只在 sim 侧有监督，GRL 可能让模型干脆不学这类几何）；(iii) 训练不稳定。正确分工是让判别器"打一场快赢的仗"。

### 7.3 双探针诊断（必须实现，是核心观测手段）

- **域探针 [已实现 2026-09-14，三通道版]**：`DomainProbe`（§2.4）在 patch / reg_domain / reg_agnostic 三通道上各挂线性头，detach 不反传，常驻训练循环监控。读数口径：**reg_domain 应高**（域信息的设计住所）、**reg_agnostic 目标 ≈ 随机（50%）**（Stage 2 GRL 的唯一清理对象）、**patch 量化登记不设硬门**（隔离优于摧毁——残留已知、可查、可披露）。
- **深度探针**：冻结 z 上训浅层深度头，监控 AbsRel/δ1.25。目标：各阶段不退化。
- 深度探针同时挂在 `z_t` 与 `ẑ_{t+1}` 上；两条曲线画在同一面板，**分岔即拔河可视化**。
- **决策程序（先量后治，禁止凭感觉加工程）**：最小配置（掩码+σ+多尺度梯度+分区探针）先跑 → reg_agnostic 探针 ≈ 随机则维持现状；可判则先检查信号泄漏路径（哪个信号头读错了通道），再考虑 Stage 2 GRL 上线或加强教师监督权重，再测。

### 7.4 评估掩码与训练掩码分离

手工规则只允许出现在**评估**侧：指标只在高置信有效像素（有效值、远离深度边缘、排除明显飞点）上计算。训练侧除"是否有限值/量程内"外无任何规则。

---

## 8. VLA 集成（Qwen3-VL-2B-Instruct）

### 8.1 接入方式（推荐主路线 A，B 为备选，C 为后续研究项）

- **A. 软 token 注入（主路线）**：世界模型 latent（256×384）经轻量 Adapter 投影到 Qwen3-VL 的 LLM embedding 空间，作为额外视觉 token 与其原生视觉 token 并列输入。Qwen3-VL 原生管线（SigLIP2 视觉塔 + merger + DeepStack）保持不动。
  > **[已裁决 2026-09-11]** Adapter 机制（VLA 侧内部事务，WAM 接口恒定 256 token）：逐 token `Linear(384→512)+GELU`（≈1×1 降维卷积）→ **space-to-depth 2×2**（≈2×2 降大小恒等卷积，无损重排）→ 8×8=64 个 token × 2048 维（恰好 = Qwen3-VL-2B hidden）→ LayerNorm。无 pooling、无 cross-attention，绝对几何信息无损；输出 8×8 网格与原生视觉 token 网格同构，MRoPE 可 1:1 对齐。实现：`src/sawvla/models/vla_adapter.py`。
- **B. 特征替代/旁路**：用世界模型 latent 替换或旁路增强 Qwen3-VL 视觉塔输出。侵入性强，仅作消融。
- **C. 世界模型规划（后续）**：VLA 提出候选动作序列，T 做 latent rollout，用预测几何/任务进展打分做 MPC 式筛选。Stage 3 稳定后再做。

可行性先例：已有工作（JEPA-VLA，2026）在单卡上把 V-JEPA2 ViT-L/16 与 Qwen3-VL-2B 双双冻结做 VLA 微调——与本项目的基模选择完全一致，说明 16GB 做 Stage 3 是可行的。

### 8.2 动作头与训练

- 第一期：行为克隆。Qwen3-VL 输出（或取末层 hidden state）→ 小型动作头（MLP 或轻量 DiT/flow-matching 头，π0 风格为可选升级），直接回归/生成动作块（action chunk，长度建议 8–16）。
- 微调策略（16GB 硬约束）：**冻结视觉塔与世界模型；LoRA（r=16~32，作用在 LLM 的 q/k/v/o）+ paged_adamw_8bit + 梯度检查点**；batch 4–8 + 梯度累积。
- 训练数据：先用 sim 域（深度完美、量大），再在 real 域上评估零样本迁移，最后少量 real 数据微调并对比。
- **核心实验**：对比"原生 Qwen3-VL 视觉特征"与"注入世界模型 latent"在 (i) sim→real 迁移成功率、(ii) 真实成功率 上的差异——这是验证"深度隔离"假设的最终裁决实验。

---

## 9. 显存预算（16GB，估算值，实测校准）

| 阶段 | 主要占用 | 估算 | 手段 |
|---|---|---|---|
| 教师离线推理 | DA-V2-L (335M)，batch8 @518px | ≈4GB | 一次性和训练解耦 |
| Stage 0 | DINOv2-S + DPT(64×64 输出)，batch32 @224px | 6–8GB | bf16 |
| Stage 1 | +T(≤100M)，8 帧 clip，batch8–16 | 8–12GB | 冻结 E、梯度检查点 |
| Stage 2 | 全模块 + 判别器 | ≤13GB | 小 lr、必要时 batch 减半累积 |
| Stage 3 | Qwen3-VL-2B bf16（≈4–5GB）+ LoRA + adapter + 动作头 | 12–14GB | 冻视觉塔、8-bit optimizer、梯度检查点、累积 |

通用规则：bf16 autocast；`flash-attn` 可用则开；梯度累积等效大 batch；显存紧张时的降载顺序：先减 batch→再降深度输出分辨率（64×64 已很低，一般不动）→再降输入分辨率（不低于 192px，避免几何细节损失）。

### 9.1 算力假设与训练时间预算

算力假设：4070 Ti Super BF16 dense 峰值 88.2 TFLOPS、显存带宽 672 GB/s；有效算力按 25–35 TFLOPS（MFU 25–40%）估算。模块级 FLOPs 推导见《隐空间与监督设计.md》§9。

| 环节 | 总时间估算 |
|---|---|
| 教师深度离线推理（~14.5 万图 `[已核实 2026-09-11，Franka]`，DA-V2-L） | ~0.5–1 h |
| 隐空间缓存（Stage 1 前置，全帧 E 前向） | <0.5 h |
| Stage 0（~100K 步，batch32） | 4–8 h |
| Stage 1（~150K 步，读缓存 latent） | 4–12 h |
| Stage 2（~80K 步，联合+GRL） | 3–8 h |
| Stage 3（Qwen3-VL-2B LoRA，40–80K 步） | 1–2 天 |

**Stage 0–2 合计约 1 天，全项目 2–3 天 GPU（教师离线推理在实测数据量下已不足 1 小时）。** 数据量前提 `[已核实 2026-09-11，Franka]`：真机 100GB / 300 success episodes / **8.8 万帧**，仿真 48GB / 319 success episodes / **5.7 万帧**，合计约 14.5 万帧；域比例 real:sim ≈ **1.6:1**（轻度不平衡，DomainBalancedSampler 仍强制 50/50）；两域同一 matched 任务（hang_cup_on_cup_holder），全库即 matched。可行前提：Stage 1 走缓存 latent（28GB 落盘，磁盘充裕）；Stage 3 严格 LoRA + 冻结；数据管线用预处理 shard + 多 worker 预取。

---

## 10. 评估指标与验收

| 类别 | 指标 | 评估集与口径 |
|---|---|---|
| 深度质量 | AbsRel、RMSE、δ<1.25 | 仅高置信有效像素（§7.4）；分域报告 |
| 表征几何含量 | 线性深度探针 AbsRel（冻结 z） | 各阶段必测，作为阶段门 |
| 域泄漏 | 域探针准确率（**patch / reg_domain / reg_agnostic 三通道分别报告**，§2.4） | reg_agnostic 目标 ≈50%；patch 残留量化披露不设硬门 |
| 动力学 | 1/4/8 步 latent smooth-L1；D(ẑ) 深度指标随步数衰减曲线 | val 集 |
| 下游 | VLA 成功率：sim、real、sim→real 零样本 | 最终裁决实验 |

**消融矩阵（论文/报告用）**：{mask-only, +σ, +教师} × {GRL on/off} × {解码损失 stop-grad on/off（验证 DINO-WM 结论在本设定是否复现）} × {patch latent / 全局向量} × {编码器初始化：DINOv2 / SigLIP2 / 随机} × {深度监督分辨率：64×64 / 112×112 / 224×224（验证降采样结论）}。

---

## 11. 工程规范

### 11.1 仓库结构

```
project/
├── configs/            # yaml：data / model / loss / stage，一切实验由 config 驱动
├── src/
│   ├── sawvla/
│   │   ├── data/       # introspection 工具、预处理、按域 dataloader 子类、
│   │   │               # latent_cache（Stage 1 缓存与 clip 数据集）
│   │   ├── models/     # encoder / transition / depth_decoder / discriminator / domain_probe / vla_adapter
│   │   ├── signals/    # 可插拔全局监督信号注册表（§2.4；trainer 零改动增删信号）
│   │   └── losses/     # metric_nll / ssi_teacher / multiscale_grad / smooth / dyn / grl
│   └── simulation/     # 自建仿真环境（IsaacLab；机器人/相机/RL 后端均可配，见其 README）
├── scripts/            # 一键脚本：prepare_data.sh / train_stage{0..3}.sh / eval_*.sh
├── docs/               # data_schema.md（§4.1 产出）、experiments.md（实验台账）
└── tests/              # 掩码逻辑、loss 数值、GRL 符号方向、分区纪律（梯度只进指定槽位）的单测
```

### 11.2 依赖基线

- Python 3.12（conda/venv）；PyTorch 2.4+ / CUDA 12.4+；`transformers`（支持 Qwen3-VL 的版本）、`peft`、`bitsandbytes`、`flash-attn`（可选）、`modelscope`（仅权重下载）、`h5py`、`webdataset` 或 `lmdb`、`wandb`、`einops`。
- 教师模型权重与 Qwen3-VL-2B 权重经 ModelScope 下载，缓存路径写入 config。

### 11.3 开发纪律（vibe coding 约定）

1. **先 introspect 再写 dataloader**（§4.1），schema 未落档前禁止写训练代码；
2. 每个 loss 配单测：掩码为空的 batch、全有效 batch、σ 不塌缩（log σ 不趋 −∞）；掩码感知降采样的正确性（无效像素不得参与池化）；
3. GRL 单测：验证梯度符号确实反转（固定判别器，E 应往"让判别器变差"方向更新）；
4. 每个阶段先 overfit-one-batch；可视化先行——RGB / 深度真值 / 教师深度 / μ / σ / 掩码 / 洞分布七联图必须出现在 wandb 首页；
5. 固定 seed、记录 git commit 与 config diff 进 wandb；
6. 数据集路径一律走 `ROBOMIND_ROOT`，禁止硬编码绝对路径。

---

## 12. 风险登记

| 风险 | 影响 | 对策 |
|---|---|---|
| 教师深度在接触区域失真 | 关键几何学错 | L_metric 度量锚在有效点纠偏；必要时换 Metric3D 类度量教师对比 |
| GRL 拔河不稳/过杀 | 几何被误伤或域残留 | λ 缓慢 ramp + 双探针监控，过杀即回调 λ |
| 16GB OOM | 训练中断 | §9 降载顺序；Stage 3 绝不全量微调 |
| sim/real 动作空间或控制频率不一致 | 转移模型学偏 | §4.1 introspection 时核对动作定义，必要时做动作重参数化 |
| 动态阶段几何被冲淡 | 深度探针退化 | stop-grad 解码损失常驻 + Stage 2 联合微调时保留 Stage 0 损失小权重 |
| 预训练 SSL 特征被深度监督冲掉 | 通用性丧失、过拟合窄域 | 防漂移锚（§2.3-E）+ SSL 通用性探针监控 |
| 解码器偷看 RGB（skip 泄漏） | 几何探针失效，方案论证崩塌 | D 仅接 z（§2.2），代码评审检查项 |
| sim/real 数据量不平衡（约 1:15） | GRL 判别器躺赢 / sim 重复过拟合 | GRL batch 强制 50/50 域均衡；sim 侧强 RGB 增广；采样比例 config 化 |
| 数据集实际 schema 与假设不符 | 管线返工 | §4.1 是第 0 号任务，先于一切 |

---

## 13. 里程碑

- **M0** 数据 introspection 完成，`docs/data_schema.md` 落档；
- **M1** 预处理缓存（教师深度、中值背景、掩码、64×64 监督目标、shard）就绪；
- **M2** Stage 0 通过验收门（深度探针达标 + mask-only 基线对比取胜 + SSL 探针不退化）；
- **M3** Stage 1 通过验收门（多步预测稳定）；
- **M4** Stage 2 通过验收门（reg_agnostic 域探针 ≈ 随机且深度探针不退化，patch 泄漏量已登记）；
- **M5** Stage 3 sim 域 BC 跑通，sim→real 零样本评估出数；
- **M6** real 域微调与最终对比实验（§8.2 核心实验）完成。

---

## 14. 参考资料

- Depth Anything V2（深度教师，主干 DINOv2）：https://github.com/DepthAnything/Depth-Anything-V2
- DINO-WM（特征空间世界模型；patch vs 全局向量、解码损失 stop-grad、frameskip、196×384 隐空间规格）：https://arxiv.org/abs/2411.04983
- V-JEPA 2（从零视频 SSL 编码器；3D-RoPE；AC predictor 300M 从零、编码器冻结）：https://ai.meta.com/research/v-jepa-2-world-model-benchmarks/
- 冻结 DINOv2 隐空间上大规模视频世界模型预训练（~60M 网页视频）：https://arxiv.org/abs/2507.19468
- 异方差不确定性加权（Kendall & Gal）：https://arxiv.org/abs/1703.04977
- Scale-and-shift invariant 深度损失（MiDaS）：https://arxiv.org/abs/1907.01341
- 多尺度梯度损失（Eigen et al.）：https://arxiv.org/abs/1406.2283
- Qwen3-VL（VL 基模）：https://github.com/QwenLM/Qwen3-VL
- RoboMIND 2.0 数据集（ModelScope）：`X-Humanoid/RoboMIND2.0-Franka-Part-1`（real）、`X-Humanoid/RoboMIND2.0-Franka-sim`（sim）；Tienkung 版已归档（2026-09-11 切换）

---

## 附：已否决方案备忘（防止实现者重新引入）

1. ❌ 手工设计深度有效性规则做训练过滤——阈值敏感、材质偏置、丢弃信息；只允许用于评估掩码与极端非法值剔除。
2. ❌ 对深度目标做低通滤波对齐域分布——洞不是高频噪声，系统性偏差是低频的，滤波杀不掉；且专杀夹爪/轮廓等关键几何。用**多尺度梯度损失**（损失侧低通）替代。
3. ❌ 手工给 sim 深度加传感器噪声——工程量大、难做完美；用**教师稠密化**替代。
4. ❌ 只靠 GRL 不做监督对齐——拔河欠收敛或过杀；GRL 只负责扫尾。
5. ❌ 固定相机下的光度一致性自监督——只有运动区域有梯度，收益低，不做。
6. ❌ 动态阶段把深度解码损失反传进 E/T——DINO-WM 消融实证有害；stop-grad。
7. ❌ VQ 离散隐空间——GRL 反传不顺、码本坍缩风险；连续 latent。
8. ❌ 冻结编码器（DINO-WM 的做法）——本项目需要 GRL 反传进 E 除域、Stage 0 深度损失塑形 E，冻结后两者无处着力；DINOv2 只作初始化，不作冻结底座。
9. ❌ VLM（CLIP/SigLIP 类语言对齐）参数作为编码器默认初始化——语义压缩丢几何信息，与世界模型的几何目标错配；仅列为消融。
10. ❌ 全局单向量隐空间 / object-centric 槽位极致压缩——前者被 DINO-WM 消融证伪，后者无法解码稠密深度；空间网格不可压，要压只压通道 d。
11. ❌ 解码器使用 encoder skip connection——深度走 RGB 捷径，z 几何探针失效。
12. ❌ 满分辨率（224×224）深度监督——超过 16×16 网格信息上限的高频内容是解码器幻觉，污染梯度；监督/输出 64×64（上限 112×112）。
13. ❌ Qwen3-VL-2B 全量微调——16GB 不可行；LoRA + 8-bit 优化器。
