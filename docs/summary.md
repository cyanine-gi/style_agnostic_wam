# Style-Agnostic WAM — 项目总览（v2）

> 真机 + 仿真视频混合训练的世界-动作模型（WAM）demo：**训练时不区分画面来自仿真还是真实**，用表征层面的设计让风格差异失效。
> v2 变更（2026-09-07）：主线收缩为世界模型；VLA/SmolVLA/LIBERO 移出 v1，仅作 future work；场景定为 Franka Panda 桌面操作（方案 A）。

## 1. 一句话定位

混合**同本体**的仿真（NVIDIA PhysicalAI-SingleArm，Isaac Sim 生成）与真机（DROID 同任务子集）视频 → 自监督学习跨域统一的 **latent action** 表征 → 同一 latent action 序列驱动世界模型分别生成仿真风格与真机风格的未来 → 用"混合 vs 单域"消融证明风格无关表征的价值。

## 2. 场景：Franka Panda 桌面操作（方案 A）

| | 仿真侧 | 真机侧 |
|---|---|---|
| 数据集 | `nvidia/PhysicalAI-Robotics-Manipulation-SingleArm`（~38k 条，Isaac Sim 生成，LeRobot 格式，CC BY 4.0） | DROID LeRobot 移植版（~92k 条全量，本项目取同任务子集，CC-BY 4.0） |
| 机器人 | Franka Panda | Franka Panda（同一本体） |
| 任务族 | 开关抽屉 / 开关柜门 / 取放 / 堆叠 | 从 86 任务族中按语言指令过滤同任务子集 |
| 格式 | LeRobot，免转换 | LeRobot 现成移植，免转换 |

**赠品数据**：`nvidia/PhysicalAI-Robotics-Manipulation-Augmented` —— 1000 条叠方块演示，同内容同时提供仿真渲染与 Cosmos Transfer 照片级增强两个版本，是"同内容、不同风格"的天然配对，用作 S2 风格不变性的热身验证。

**备选（不进主线）**：AgiBot World + Genie Sim 3.0（官方 sim-real 对应，但双臂场景重、CC BY-NC-SA、16GB 吃力）；BridgeData V2（WidowX 本体 Isaac 生态无现成支持，可作第三域扩展）。

## 3. 为什么"风格无关"可行（技术依据）

| 路线 | 依据 | 本项目采用 |
|---|---|---|
| Latent action（LAPA/UniVLA/AdaWorld） | 从帧间变化自监督学动作，与外观解耦；AdaWorld 证明 latent action 上下文不变、可跨场景迁移 | ✅ 主线 |
| 潜空间世界模型（V-JEPA 2-AC） | 不重建像素，对风格天然鲁棒 | 参考 |
| DreamGen 式像素混合 | 真机:生成=1:1 混合训练，行业验证有效 | ✅ 混合配方（WAM 训练侧） |

**Demo 卖点图**：同一段 latent action 序列，分别接仿真首帧与真机首帧自回归生成两段未来视频 —— 直观展示表征层的动作语义与画面风格解耦。
（注意：卖点图本身只证明模型保留首帧风格；风格无关的**定量证据**是 S2 的跨域 latent action 匹配率，两者必须一起交付。）

## 4. 系统流水线

```
 数据层   PhysicalAI-SingleArm(仿真) + DROID同任务子集(真机) + Augmented(风格配对热身) → 统一 LeRobot 格式（不带域标签）
 表征层   Latent-Action VQ-VAE（帧对 → 离散动作码，~30M，混合数据自监督）
 WAM     离散帧 token + latent action → 自回归 Transformer（~200M）→ 未来帧 token → 视频 tokenizer 解码双风格未来
 评测层   表征级：跨域匹配率 / 线性探测；生成级：分域 FVD / 未来帧 LPIPS；消融：仅仿真 / 仅真机 / 1:1 混合
 演示层   Gradio：同一 latent action → 双风格未来视频
```

## 5. 仓库结构

```
style_agnostic_wam/
├── AGENTS.md
├── docs/                    # 本目录；summary 递归索引全部文档
├── configs/
├── scripts/                 # s0_~s4_ 阶段入口
└── src/sawvla/
    ├── data/       # 下载、任务过滤、LeRobot 统一、混合采样器
    ├── latent/     # latent action VQ-VAE
    ├── wam/        # 自回归世界-动作模型
    ├── train/      # 训练入口
    ├── eval/       # 表征级 + 生成级评测
    └── demo/       # Gradio：同一动作生成两种风格未来
```

## 6. 文档索引（递归引用）

| 文档 | 什么时候读 |
|---|---|
| [stages.md](stages.md) | 规划当前阶段、找脚本入口、查验收标准 |
| [data_preparation.md](data_preparation.md) | 下载/过滤/混合任何数据之前 |
| [pretrained_model_dependency.md](pretrained_model_dependency.md) | 需要权重、评估显存、查许可之前 |
| [architecture.md](architecture.md) | 写/改模型、数据模块代码之前 |

## 7. 硬件与存储预算

- 16GB 可行：latent action VQ-VAE（~30M）与 WAM（~200M）全参自训；视频 tokenizer 仅推理。
- 磁盘：数据 ~150GB + 权重 ~20GB + checkpoint ~200GB（v1 砍掉 VLA 后大幅缩水）。

## 8. 工期总览（7 周，单人）

| 周 | 阶段 | 里程碑 |
|---|---|---|
| W1 | S0 环境 | LeRobot 栈 + IsaacLab + tokenizer 自检通过 |
| W1–W2 | S1 数据 | 三源数据统一格式，混合采样器就绪 |
| W2–W4 | S2 latent action | VQ-VAE 训完；Augmented 热身 + DROID↔Sim 跨域匹配自检通过 |
| W4–W6 | S3 WAM | 双风格生成卖点图 + 三组消融训完 |
| W6–W7 | S4 评测+发布 | 指标表、Gradio demo、开源发布 |

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| latent action 学不出跨域一致性 | 先用 Augmented 风格对做热身定位问题；加域对抗头/风格随机化兜底 |
| DROID 同任务子集量不足 | 放宽任务族关键词；放宽到全 DROID 子采样（世界模型不依赖任务标签） |
| WAM 生成质量差（MSE 塌缩等） | 帧 token 用离散 tokenizer + CE 损失（见 architecture.md），不用连续 latent + MSE |
| 16GB 跑不动某配置 | stages.md 每步都有降配预案（分辨率/batch/帧数） |

## 10. 状态追踪

| 阶段 | 状态 | 备注 |
|---|---|---|
| S0–S4 | ⬜ 未开始 | v1 创建：2026-09-07；v2 重写（方案 A）：2026-09-07 |
