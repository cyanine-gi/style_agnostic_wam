# 端到端信息流设计：动作注入、位置编码、因果注意力、VLA 架构形态

> 本文档说明世界模型与 Qwen3-VL 之间的端到端信息流，回答四个问题：
> (1) 动作怎么注入世界模型；(2) Qwen3-VL 与 DINOv2 的位置编码是否统一、如何统一；(3) 全链路因果注意力怎么设计；(4) VLA 侧采用什么架构形态——**双流/双塔并存**（§6 决策与机制，§7 同类框架对比依据）。
> 姊妹文档：`guideline.md`（训练流程与工程约束）、"latent_space_and_supervision_design.md"（隐空间/解码器/算力）。冲突时以较新者为准。发现冲突第一时间告知用户并询问意见.

---

## 0. 总图

```
                        【世界模型侧】                                   【VLA 侧】
 RGB_t → E(DINOv2-S/14, 帧内双向) → z_t ─┐
 RGB_{t-1} → E ─────────────────→ z_{t-1} ─┤   Adapter(Linear→GELU→space-to-depth 2×2→LN)
                                          ├──► WM token(8×8=64, 手工 MRoPE id) ──┐
 action chunk → MLP → 逐步 action token ──┐  │                                   │
 proprio → MLP → proprio token ──────────┤  │   文本指令(原生分词)               │
 z_t ────────────────────────────────────┴──► T(block-causal) → ẑ_{t+1..k}      │
                                                                            ▼
        D(仅接 z/ẑ, 卷积上采样 64×64) ◄── z_t, ẑ(stop-grad 隔离)   Qwen3-VL-2B(原生因果 LM)
              │                                                     │  原生视觉 token(8×8, merger)
              └─► 深度 (μ,σ) → 训练监督/探针/可视化                  │  + WM token(8×8, 注入)
                                                                    ▼
                                                            动作 query → 动作块
```

---

## 1. 动作注入世界模型

### 1.1 第一原则：动作只进 T，不进 E

z 必须是**纯观测表征**。动作若进 E，z_t 就携带了未来信息，Stage 0 的静态重建语义、深度探针、域探针全部失去解释性。推理时 E 只看当前帧 RGB。

### 1.2 注入结构

- **数据侧**：动作向量维度 `[已核实 2026-09-11，Franka]` = **16**（双臂 7DoF 关节 + 双夹爪，action=master 指令、proprio=puppet 实测；夹爪方向两域一致：高=抓握）。归一化统计量只从 train split 计算；**sim/real 的动作量纲与控制频率必须在 introspection 时核对**（Franka 已核对：关节同为 rad、30Hz 对齐序列；夹爪 real 归一化 [0,1] / sim 连续 [0,~0.16]，归一化统计桥接，见 dataloader.md §12.5），不一致则先做动作重参数化再统一归一化（域差异不能在动作通道上混进 T）。
- **token 化**：action chunk 共 k 步，每步动作向量经 MLP(action_dim→384) 升为 **1 个 action token**（k 步 = k 个 token，每步一个，不用固定数压缩——逐步 token 的因果语义干净）；本体感向量 → 1 个 proprio token。
- **条件方式**：**token 拼接进 T 的自注意力序列**为主方案（每个 patch 可通过注意力决定自己受动作影响的程度，空间选择性天然具备）；AdaLN 全局调制为消融项。
- **位置编码**：action/proprio token **只加时间维位置编码**（各自 step 索引），不加空间编码——它们是全局量，不绑定 uv（V-JEPA 2-AC 做法）。
- **输出**：T 取 z 位对应的 256 个 token → Linear(d→d) + LayerNorm → ẑ；输出层小初始化，使初期 ẑ≈z_t 近恒等起步。

### 1.3 训练与推理的一致性

- 训练：block-causal 一次前向并行算 k 步损失（见 §3），frameskip ∈ {1,2,4,8} 随机采样；
- 推理 rollout：ẑ 回喂作为下一步输入，动作来自策略输出或 MPC 候选，全程不再触碰 RGB。

---

## 2. 位置编码体系：不统一编码函数，统一坐标系语义

### 2.1 三层各用各的（互不冲突）

| 层 | 网络 | 位置编码 | 作用对象 |
|---|---|---|---|
| E | DINOv2-S/14 | 绝对 sincos（可插值），2D | patch token 输入，绑定 14×14 像素块 |
| T | 自研 Transformer | patch token 用 2D-RoPE；动作/本体感 token 只加时间维编码；多步加帧索引 embedding | 序列内所有 token |
| LLM | Qwen3-VL-2B | MRoPE，每 token 一个 (t, h, w) 三元组 | 文本/原生视觉/WM token 统一编排 |

**"Qwen3-VL 和 DINOv2 的位置编码统一吗？"——编码函数不统一，也不需要统一。** 一个是绝对加式、一个是旋转式，各自活在不同网络里，互不影响。必须统一的只有一件事：**坐标系语义**——z 的 token (i,j) 与原生视觉 token 在 (h, w) 坐标上指向同一个图像区域。

### 2.2 统一做法：手工构造 WM token 的 MRoPE position_ids

- WM latent 先经 VLA 侧 adapter（逐 token Linear+GELU → space-to-depth 2×2，见 `guideline.md` §8.1 已裁决项）输出 8×8=64 个 token，与原生视觉 token 的 8×8 网格 **1:1 同构**；`z_t` 注入 token (i,j) 的 MRoPE 坐标：t = 当前帧索引，(h, w) = adapter 输出的 8×8 网格位置，与对应原生视觉 token 完全同坐标；
- 历史帧 `z_{t−1}`：同 h/w，t 减 1；未来 rollout `ẑ_{t+k}`：t 加 k；
- 文本 token 保持 Qwen3-VL 原生惯例（t=h=w=序列位置），不改。

### 2.3 为什么这样有效

RoPE 的注意力项只依赖**相对位置** (Δt, Δh, Δw)。同一物理位置的"原生视觉 token ↔ WM token"相对位置 ≈ 0，跨模态注意力天然获得**局部对齐先验**——两路流的对应关系不需要从数据里从零学。这把世界模型侧的 uv 绑定价值延伸到了 VLA 侧。

### 2.4 工程注意

- HF 实现的 MRoPE position_ids 自动计算只认原生图像/文本；注入 token 必须**自定义 collator** 手工构造 position_ids（以及配套的 labels 掩码，见 §3.3）；
- Adapter 本身（Linear/Perceiver）不携带位置信息，位置全部经由 MRoPE id 进入 LLM——不要在 Adapter 里再 embed 一份位置；
- 消融项：WM token 用正确 (t,h,w) vs 打乱 h/w，验证坐标对齐先验的收益。

---

## 3. 因果注意力设计

### 3.1 三个网络，三种掩码

| 网络 | 掩码 | 理由 |
|---|---|---|
| E（DINOv2） | 帧内**全双向** | 纯观测编码，单帧内无因果概念 |
| T | **block-causal**：帧内（z_t 的 256 token 之间）双向；跨时间严格因果 | 见 §3.2 |
| Qwen3-VL | **原生自回归因果**，注入顺序即时间顺序 | 见 §3.3 |

### 3.2 T 的 block-causal

- 第 j 步的预测只能 attend：z_t、action token a_{1..j}、已预测的 ẑ_{<j}；**禁止看 a_{>j}**；
- 为什么：若训练时第 1 步预测能看到 a_3，模型会学会用未来动作作弊，而推理 rollout 时未来动作不存在——训练-推理失配，多步展开必崩；
- 实现：block-causal mask，一次前向并行计算 k 步损失（V-JEPA 2-AC 式），而不是自回归循环 k 次前向。

### 3.3 LLM 侧

- 保持 Qwen3-VL 原生全因果 LM 掩码，**不要**改成 prefix 双向（会破坏预训练权重的分布假设）；
- 注入顺序 = 时间顺序：文本指令 → 原生视觉 token → WM token（z_{t−1} 在前，z_t 在后）→ 动作 query；
- 视觉/WM token 是上下文：参与注意力、但 labels 屏蔽不算损失；动作 query 及其后内容 attend 它们；
- 若未来启用规划模式（送 ẑ_{t+1..k}），rollout token 按 t 递增排在 z_t 之后、动作 query 之前。

### 3.4 因果泄漏单元测试（必须实现）

构造两个仅未来动作不同（a_{t+2} 改、其余全同）的输入，断言 ẑ_{t+1} 输出逐位不变。任何掩码 bug 都会被这个测试当场抓住。LLM 侧同理：改动 WM token 之后的文本，断言 WM token 位置的隐状态不变。

---

## 4. 端到端信息流连通性检查表

| 边 | 数据 | 可导性 | 冻结状态 |
|---|---|---|---|
| RGB → E → z | 图像 → 256×384 | 训练期可导 | Stage 0 训练 / Stage 1 冻结 / Stage 2 小 lr / Stage 3 冻结 |
| action → MLP → T | 动作向量 | 可导 | 始终训练（T 侧） |
| z_t → T → ẑ | latent | 可导（对 T） | T 从零训练 |
| ẑ/z → D → (μ,σ) | latent → 64×64 深度 | **stop-grad 隔离 E/T**（仅 Stage 0 反传进 E） | D 始终训练 |
| z → 域判别器 | latent → 域标签 | GRL 反转进 E、T | Stage 2 起 |
| z → Adapter → LLM | latent → 软 token | E 冻结，梯度止于 Adapter | Stage 3：只训 Adapter + LoRA + 动作头 |
| EMA 副本 → T 的目标 | latent（目标） | 不可导 | 永不训练 |

**不变量**：推理时从 RGB 到动作输出的通路上，不存在任何依赖深度输入、未来动作、未来帧的边。

---

## 5. 工程检查清单（coding agent 用）

1. 自定义 collator：手工构造注入 token 的 MRoPE position_ids 与 labels 掩码（只算动作 token 损失）；
2. Adapter 输出初始化缩放到原生视觉 token embedding 的范数量级；
3. 因果泄漏单测（§3.4）进 CI；
4. 反事实置零评估脚本：WM token 置零后动作误差应显著变差，否则说明 WM 流没被用上（多模态融合的"强模态碾压"翻车模式）；
5. 动作归一化统计量只来自 train split，sim/real 动作量纲一致性在 introspection 阶段核对；
6. wandb 固定面板：双探针曲线、rollout 深度可视化、反事实置零对比。

---

## 6. Qwen3-VL 视觉流：双塔并存决策与模态 dropout 课程

**结论已定：采用双流/双塔方案——LLM 主干 + SigLIP2 视觉流 + WM 流三者全部保留，双塔并存为部署形态**（同类框架依据见 §7）。本节只保留两个配套机制：防"强模态碾压"的模态 dropout 课程（§6.2），以及转为测量手段的原三臂实验（§6.3）。

### 6.1 组件拆分与决策

| 组件 | 角色 | 决策 |
|---|---|---|
| LLM 主干 | 语言理解 + 推理基质 | **必须保留**：多任务指令需要真正的语言理解，2B 的预训练语言知识无法在 ~30 万帧机器人数据上重训 |
| SigLIP2 视觉流（+merger/DeepStack） | appearance 通道、语言对齐的 grounding 脚手架 | **保留**：与 WM 流双塔并存；域信息泄漏靠 WM 侧 GRL/双探针治理（guideline_v2 §7），不靠拆视觉塔 |

### 6.2 模态 dropout 课程（训练期机制）

双塔并存的最大风险是**强模态碾压**：语言对齐的 SigLIP2 流是捷径，LLM 可能全程忽略 WM token，双塔形同虚设。模态 dropout 是反制手段：

- **机制：block 级 modality dropout**——整块原生视觉 token 以概率 p 从输入序列移除，p 从 0.3 退火至峰值 0.7~0.9（分段或余弦，config 化）。**禁止连续加权（α·embedding）**：连续缩放制造分布外的中间态输入，dropout 让网络永远只见干净的子集。WM block 保留 ~0.1 的对称 dropout（标准 modality dropout）。
- **grounding 脚手架角色**：SigLIP2 是语言对齐特征，"指令名词 ↔ 图像区域"的对应近乎预训练白送；WM token（DINOv2 特征空间）**不是语言对齐的**，冷启动 grounding 难。训练早期视觉流当脚手架教会 LLM grounding，中期高 p 强迫 WM 通路独立承载同样对应。
- **收尾校准**：末段训练 p 回落至 ~0.2，使训练末分布与部署分布（双通路俱全）一致——dropout 是正则与测量手段，不改变双塔部署形态。
- **对称收益**：视觉流被周期性丢弃时 WM 通路被迫独立工作，与反事实置零评估（§5 第 4 条）闭环；同时免费获得 WM-only 降级模式（视觉链路故障/极端外观域偏移时的容错备份，见 §6.4）。

### 6.3 三臂测量（消融与分析，不再是形态决策门）

最终形态已定（双塔并存），原三臂对比转为**测量实验**（同一 BC 数据与评测协议）：

| 臂 | 视觉流 | WM token | 测什么 |
|---|---|---|---|
| (i) | 有 | 有 | **部署形态**，性能上限 |
| (ii) | 无 | 有 | WM-only 下界：视觉流缺席时系统还剩多少能力（降级模式可行性 + 域隔离上限） |
| (iii) | 有 | 无 | 消融对照：WM token 的边际贡献 |

读法：(i)−(iii) = WM 流的边际价值（反事实置零评估的严格版）；(i)−(ii) = **外观对控制的边际价值**，直接量化项目核心假设"任务执行应与 RGB 外观质量无关"，是论文级测量结果。若 (ii)≪(i)，说明 WM 通路 grounding 不足，回去调 dropout 课程与 adapter 对齐阶段——**不**回退双塔决策。

### 6.4 部署形态

- **标准形态：双塔俱全**。视觉塔加载（~0.6GB 显存，预算内）；LLM 主干在环，靠 action chunk（8~16 步/次推理）把 LLM 调用降到 2~4Hz；2B bf16 单次前向 ~600 token ≈ 2~3 TFLOPs，4070 Ti Super 可承受；
- **降级形态：WM-only**。dropout 课程保证视觉塔缺席时策略仍可用——用于视觉链路故障、强外观域偏移（夜间/污损/遮挡相机）等场景；切换只是推理时丢弃视觉 token 块，无需改权重；
- 延迟仍敏感：蒸馏到小 policy（WM latent → 小动作头）作端侧形态，VLM 只当训练期教师。

---

## 7. 同类框架对比：双流/双塔方案的依据

### 7.1 两大流派

| 流派 | 做法 | 代表工作 |
|---|---|---|
| 统一单塔 | 一个骨干同时学"预测未来帧 token"与"动作 token"，世界建模是同一 token 流上的辅助目标 | WorldVLA、RynnVLA-002、GR-1/GR-2、UniVLA、UP-VLA、UVA/UWM |
| 双流/模块化 | 独立 WM 编码器 + 预测器，WM latent 以 token 形式注入 VLA | **VLA-JEPA**、CoWVLA、DreamVLA；旁证：OpenVLA 双视觉编码器、π0/GR00T 动作专家权重级分离 |

统一派的两个硬伤（对本项目尤其关键）：

1. **任务干扰实测存在**：WorldVLA 自身消融显示，只做世界模型预训练时视觉生成成功率 85.5%，与动作模型联合训练后掉到 66.5%——动作目标损伤了世界建模能力。统一架构里两个目标抢同一组参数；而本项目要对 WM 隐空间做 GRL/双探针等精细控制，参数共享会让这些手段失去作用对象。
2. **算力门槛**：图像 token 化（VQ-GAN）+ 视频生成式预训练的成本面向集群设计，不在单卡 4070 Ti Super 量级。

### 7.2 双流派的关键先例

- **VLA-JEPA（arXiv 2602.10098，与本方案最接近）**：Qwen3-VL（2B/4B/8B）骨干 + V-JEPA2 ViT-L 世界模型（随机初始化预测器预测未来 latent），⟨latent⟩ 潜变量 token（每帧 K=24）拼进 LLM 输入序列，DiT-B 扩散动作头；其做法是**用 WM latent 直接替换 LLM 的原生视觉 token**。Leakage-free 设计：未来帧只作监督目标、从不作输入。
- **CoWVLA / Chain of World（arXiv 2603.03195）**：独立视频编码器把片段分解为运动/结构 latent，VLM 在 latent 上做推理并预测末帧——同样是"独立编码器 + LLM 消化 WM token"的双流结构。
- **DreamVLA（arXiv 2507.04447，NeurIPS 2025）**：先预测紧凑世界知识（动态区域/深度/语义特征）再条件化动作，真机 76.7% 成功率。**其消融对本方案有警示价值**：深度/高维特征等辅助监督若与动作梯度在同一骨干上直接竞争，反而掉点——本方案的 Stage 隔离（Stage 0/1 训 WM、Stage 3 冻结 E、深度损失 stop-grad）正是规避这一干扰的设计。
- **OpenVLA（arXiv 2406.09246）**：非 WM 工作，但证明一个 LLM 同时消化两路异质视觉流（DINOv2+SigLIP 通道拼接）可行，其中 DINOv2 一路专补空间推理。

### 7.3 与 VLA-JEPA 的差异（有意为之）

| 维度 | VLA-JEPA / CoWVLA | 双流/双塔方案（本文档） | 差异理由 |
|---|---|---|---|
| WM 编码器 | 预训练 V-JEPA2，**冻结** | DINOv2-S/14 初始化、**可微调** | 深度监督 + GRL 域对抗都要求梯度进 E，冻结做不到域隔离 |
| 注入 LLM 的 token | 每帧 24 个潜变量 token，放弃显式空间结构 | WAM 接口恒定 256 patch token，VLA 侧 adapter space-to-depth 为 8×8=64 token + 手工 MRoPE 对齐（已裁决 2026-09-11） | 保 uv 结构是稠密深度解码的前提；压缩属 VLA 内部事务 |
| WM 训练数据 | 海量公开视频 | ~30 万帧机器人数据（real 217GB + sim 11.5GB） | 量级差约 100 倍 → 必须分 Stage + 离线 latent 缓存（guideline_v2 §6） |
| 视觉流 | 替换原生视觉 token（单流） | **双塔并存** + 模态 dropout + vision on/off 测量 | "外观捷径"假设需要对照测量，且视觉流是 grounding 脚手架 |
| 域处理 | 无 | GRL + 双探针（guideline_v2 §7） | sim/real 域隔离是本项目核心问题 |

### 7.4 定位结论

双流/双塔不是自创孤例，而是 latent 系 WM+VLA 的正经路线；VLA-JEPA 已替"仅 WM 臂"做过可行性验证。本项目相对文献的增量在四点：双塔并存（而非替换）、模态 dropout 课程、vision on/off 差距的量化测量、面向 sim/real 的域隔离治理（GRL + 双探针）——统一派 66.5% vs 85.5% 的干扰数字，正是"必须解耦"的直接证据。

---

## 参考

- V-JEPA 2-AC（action/proprio token 仅时间维位置编码、block-causal 预测器）：https://ai.meta.com/research/v-jepa-2-world-model-benchmarks/
- DINO-WM（frameskip、预测器结构、patch 表征）：https://arxiv.org/abs/2411.04983
- Qwen3-VL（MRoPE、merger、DeepStack）：https://github.com/QwenLM/Qwen3-VL
- VLA-JEPA（Qwen3-VL 骨干 + V-JEPA2 世界模型 + latent token 注入 LLM，§7.2/§7.3）：https://arxiv.org/abs/2602.10098
- CoWVLA / Chain of World（独立视频编码器 + VLM latent 推理，§7.2）：https://arxiv.org/abs/2603.03195
- DreamVLA（世界知识预测→动作；深度辅助监督与动作梯度干扰的警示，§7.2）：https://arxiv.org/abs/2507.04447
- OpenVLA（DINOv2+SigLIP 双视觉编码器先例，§7.2）：https://arxiv.org/abs/2406.09246
- WorldVLA（统一单塔路线；任务干扰消融 66.5% vs 85.5%，§7.1）：https://arxiv.org/abs/2506.21539
