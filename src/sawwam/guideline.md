# sawwam 设计纲领（v1，2026-09-16）

> 相机条件化 + slot 隐式场景表示的世界模型。本目录（src/sawwam）是与
> src/sawvla 并行的新代码线；sawvla（2D patch token 世界模型，stage 0–2）
> **保持不动、继续可用**，本文档是 sawwam 的唯一权威设计依据。
>
> 需求原点（2026-09-16 用户提出）：世界模型应能接受**原始图像 + 对应相机
> 内参、外参**作为输入；E 编码出的 latent **与相机位姿无关**，隐式描述
> 客观世界——"空间里有几个东西、都在哪里、受力后怎么运动"，不依赖图像
> 局部性、不要显式 pointmap；D 解码时再注入（另一组）内外参，完成
> 同视角/跨视角深度解码。

---

## 0. 裁决记录（2026-09-16，全部经用户确认）

1. **latent 形态**：隐式 slot 集合（物体中心），**显式 pointmap 方案被否决**
   （用户理由：影响下游 VLA；要隐式表达"几个东西、在哪里、受力怎么动"）。
2. **相机内外参必须进 E 的输入**；E 的 latent 输出不携带视角信息；
   D 解码时注入目标相机内外参（同视角/跨视角统一）。
3. **相机条件载体**：全局相机 token（13 维归一化内外参 → MLP），
   逐像素/逐 patch Plücker 射线图否决为冗余（机位恒定场景下信息
   等价于 token + 位置编码）；保留为 v2 升级件（若视角无关化效果不佳）。
4. **参考系**：机器人基座系（sim 有 base_to_robot_transformation；桌子在
   基座系中位置固定已知）。
5. **绑定机制**：DETR 式可学习 query 单遍 cross-attn（slot attention 的
   迭代/GRU 形式不采用），但 **softmax 在 query（slot）轴上**——slot 竞争
   patch 的解释权，恢复竞争式划分偏置，防小数据绑定崩溃。
6. **跨帧身份**：SAVi 式预测-修正——编码新帧时 query 由上一帧 slot 投影
   初始化，再被当前帧证据修正；展开想象时只有预测相。
7. **动力学预测残差**：T 回归 Δ = z_t − z_{t−1}，静止默认 Δ≈0。
8. **slot 数 K=32 起步，必须可配置**。
9. **E backbone**：冻结与微调两条路都要支持，**先冻结**。
10. **real 侧相机参数**：内参用 RealSense D455 名义值（fx≈fy≈634 @1280×720，
    已实测 RoboMIND-sim 内参即此值）；外参用每批次粗先验（0520 批 =
    src/simulation/config/simulation.yaml 中手调机位）；条件 dropout 兜底；
    0821 批（11 集）随其处置暂缓。
11. **腕部相机不做**（用户裁决：多余）。
12. **跨视角数据来源**：RoboMIND-sim（6 机位 + 精确内外参 + 稠密深度）
    **+ 自家 Isaac Lab 仿真**（管线可控；相机位姿随机化 + K/T 逐集落盘是
    **硬前提**，固定机位下所有视角无关压力失效）。
13. **新代码进 src/sawwam，不动 sawvla 老代码**；本文档替代在
    docs/guideline.md 上打补丁的做法。

## 1. 背景与动机

sawvla 线的 latent 是 2D 图像 patch 网格（16×16 token，token↔图像局部性
绑定），视角信息作为 nuisance 隐式混在特征里，靠 stage 2 GRL 清理域信息。
两域默认相机机位系统性不一致（dataloader.md §12.9：real 斜视 / sim 近顶视）
进一步说明：视角是必须显式处理的变量。

sawwam 换一个根本思路：**把相机从隐空间里拿出来，作为显式输入**；latent
改为**物体中心的无序 slot 集合**，天然不携带图像局部性，视角无关性由
训练压力隐式获得（见 §4）。下游 VLA 的接口从 256 patch token 变为
"场景物体 slot 集合"，与本项目 style-agnostic 目标同构。

技术参照家族：slot 绑定与视频物体中心表示（Slot Attention / SAVi /
SlotFormer / DINOSAUR）；slot 隐式场 + 相机条件渲染（GIRAFFE / uORF）。

## 2. 架构

```
                    相机 token (13维, 基座系)
                          │（融入 patch 特征）
RGB(224×224) ──► DINOv2-S backbone ──► 256 patch 特征 ──► 绑定模块
（v1 冻结，预留微调开关）                    （DETR 式, K=32 query,
                                             softmax 在 query 轴,
                                             query 由上一帧 slot 投影初始化）
                                                        │
              latent = 32 slot（无序集合，无位置编码） + 4 register
                                                        │
              ┌─────────────────────────────────────────┤
              ▼                                         ▼
        T（block-causal transformer）              D（slot 渲染器）
        block = 32 slot + 4 reg + 2 cond           query = 目标相机 uv 网格
        预测残差 Δ，ẑ = z + Δ                        + 目标相机 token
        register 递推 / free-run 沿用 sawvla          cross-attn over slots
                                                   → 64×64 深度
```

### 2.1 相机 token（13 维）

- 内参 4 维：fx/W, fy/H, cx/W, cy/H（按图像尺寸归一化）；
- 位姿 9 维：旋转 6D 连续表示（避开四元数双覆盖）+ 平移 3 维（按场景尺度
  归一化，桌面高约 0.75m）；
- 参考系：机器人基座系；
- **预处理折叠**：letterbox（1280×720 → 224×126 + 上下填黑）必须折进 K：
  fx/fy 乘 224/1280、cx 同比例、cy 同比例再加黑边偏移——喂模型的是
  "预处理后图像"的等效内参；
- 经 MLP 升维后融入 patch 特征（加性/拼接由实现定）。

### 2.2 绑定模块（DETR 式，竞争归一化）

- K 个可学习 query（K=32，配置项），单遍 cross-attn over patch 特征；
- **softmax 在 query 轴**：每个 patch 的注意力质量在 32 个 slot 间归一化，
  slot 间形成竞争划分（防绑定崩溃的关键，见 §5 风险 1）；
- 修正相：query 输入 = 基础可学习 query + 上一帧 slot 的投影
  （SAVi 式身份保持）；首帧用基础 query；
- 输出 K 个 slot 向量，无序、无位置编码。

### 2.3 T（动力学）

- 沿用 sawvla 的 block-causal transformer + register 递推 + free-run 展开；
  block = 32 slot + 4 register + 2 cond（action, proprio），token 数从
  262 降到 ~38，计算量大降；
- **残差头**：预测 Δ，ẑ_t = z_{t−1} + Δ；静止背景/物体零压力；
- slot 上**不打 RoPE**（集合无空间结构）；
- T 跨所有视角共享权重 ⇒ 动力学在世界系才视角无关 ⇒ T 本身是
  视角不变性的压力源（§4 第 3 股）。

### 2.4 D（slot 渲染器）

- query = 目标相机 64×64 uv 网格 + 目标相机 token；对 32 slot 做
  cross-attn 出视差（z-depth，RoboMIND 约定毫米、65535 哨兵沿用）；
- 同视角与跨视角解码统一于此；小网络，无显式重投影结构（保持学习能力，
  允许遮挡幻觉）；
- 深度损失组合沿用 sawvla（metric/teacher/grad/smooth 分项 + 掩码）。

## 3. Register 分区（沿用 sawvla）

4 个 register：[0,1] = reg_domain（域/风格槽位），[2,3] = reg_agnostic
（域无关槽位）；信号头与（stage 2 复用时）GRL 仍读 reg_agnostic；
CLS 继续闲置。slot 集合承载物体/场景内容，与 register 分工不变。

## 4. 视角无关性的三股隐式压力（无显式几何监督）

1. **跨视角解码瓶颈**：latent + 相机 B 参数 → B 视角深度。相机随机化数据
   上，slot 若不隐式编码物体 3D 位置，任意目标视角无法渲染正确深度。
2. **跨视角 slot 集合一致性**（sim）：同一场景两视角分别编码，两份 slot
   集合做 Hungarian 匹配后逐对拉近——监督"内容一致"，不问坐标。
3. **共享动力学瓶颈**：T 跨视角共享；视角相关的 slot 会迫使 T 每视角
   学一套动力学，损失惩罚这一点。

**命门**：以上三股压力在机位固定时全部失效（背机位即可通过所有损失）。
实测 RoboMIND-sim 全库内参偏差 1e-13、外参 1.4cm ⇒ 相机随机化只能来自
自家 Isaac Lab 环境（§6.3）。

## 5. 风险清单

1. **绑定崩溃/不绑定**（头号风险）：多 query 注意同一块、或一个 slot
   吃全部。缓解：query 轴 softmax 竞争；稠密逐像素深度监督（每帧 4096
   监督点，远密于 DETR 的稀疏框标签——小数据可训性的主要依据）；
   监测：定期可视化 32 个 slot 的注意力归属图，肉眼立判。
2. **数据规模**：参照系（SAVi/DINOSAUR 在 5k–100k 条简单场景 clip 可训）；
   我们 619 集 / 8.4 万帧在量级内；不够时自家仿真一个晚上可再产几百集，
   上限在自己手里。
3. **slot 跨帧身份漂移**：修正相（query 从上一帧 slot 初始化）构造性
   保持；探针：深度解码归属图上色追踪跨帧对应。
4. **"在哪里"无绝对标尺**：无显式坐标监督，slot 的 3D 位置是隐式编码；
   加只读探针头回归物体基座系坐标做验证，不进损失。
5. **real 侧二等公民**（外参未解前）：real 只吃同视角深度损失，跨视角
   与集合一致性仅 sim；视角无关性主要在随机化 sim 学成，real 靠
   域拉近（stage 2 GRL 逻辑复用）。机械臂 PnP 标定（real 每帧有
   puppet 关节角 + Franka 运动学，臂在画面中可反解外参）为可选子项目，
   暂不启动。
6. **K 固定 vs 物体数可变**：多余 slot 学空态，实践中成熟。
7. **静态机位红利**：相机固定 + 背景静止 ⇒ 背景动力学免费可预测，
   slot 容量自然让给会动的物体——对本 setup 是有利偏置。

## 6. 数据

### 6.1 RoboMIND real（300 集）

- 4 路相机（front/left/right/top），有深度（洞 + 65535 哨兵），
  **无内外参**；
- 内参：D455 名义值；外参：每批次粗先验（0520 = simulation.yaml
  手调机位）；条件 dropout（~10% 丢相机 token）兜底；
- 仅参与同视角深度解码 + 动力学；跨视角损失在外参解决前不做。

### 6.2 RoboMIND sim（319 集）

- 6 路相机，精确内外参逐集存储，稠密深度；机位全库恒定（不能学
  相机条件的泛化，但 front→top 等跨视角监督对本身真实可用）；
- 用途：跨视角深度解码 + 跨视角 slot 集合一致性。

### 6.3 自家 Isaac Lab 环境（src/simulation，硬前提工程）

- cameras 配置加**位姿/焦距随机化范围**（建议 pos ±15cm、look_at ±5cm、
  focal 16–20mm，量级可调）；
- 采集时把精确 K、T **逐集写进 HDF5**（无腕部相机，固定机位每集一份）；
- 可摆 2–3 路相机增加视角多样性；深度语义沿用
  distance_to_image_plane（z-depth，m→mm）。

## 7. 阶段计划

| 阶段 | 内容 | 复用 sawvla |
|---|---|---|
| **w-stage 0** | backbone 冻结 + 绑定模块/D 从零训；损失 = 同视角深度(real+sim) + 跨视角深度(sim) + 集合一致性(sim) | E backbone、图像预处理、深度损失、探针框架 |
| **w-stage 1** | T 换 slot 输入 + 残差头；free-run 展开、register 递推、Option A 梯度语义沿用 | T 框架、训练循环、checkpoint 机制 |
| **w-stage 2** | GRL 清理 reg_agnostic 域信息，逻辑不变 | discriminator、GRL、域均衡采样 |
| **w-stage 3** | VLA adapter 接口 = slot 集合（物体中心状态） | adapter 概念 |

每 1000 步完整 checkpoint（模型 + 全部优化器状态 + RNG，--resume 一键
续训）等工程约定沿用 sawvla 既有实现。

## 8. 与 sawvla 的关系

- sawvla 代码与文档（docs/guideline.md）**不再改动**，其 stage 0–2 产出
  （checkpoint、latent 缓存、训练脚本）保持可用，作为 2D patch 路线的
  基线与对照；
- sawwam 的 stage 2 若跑出结果，与 sawvla 的对比即"2D patch latent vs
  slot 隐式场景 latent"的路线对照实验；
- 共用：数据集读取（dataset.py / stage0_dataloader.py，按 sawwam 需要
  扩展相机参数字段）、图像预处理、信号注册表概念、checkpoint/日志/
  可视化工程约定。

## 9. 待办与遗留

- [ ] Isaac Lab 环境相机随机化 + K/T 落盘（§6.3，硬前提，最先做）
- [ ] 数据管线扩展：dataset/dataloader 输出相机 token 所需 13 维参数
      （sim 读文件、real 用名义值 + 粗先验、条件 dropout）
- [ ] w-stage 0 训练脚本（含 slot 归属图可视化、绑定崩溃监测）
- [ ] 跨帧 slot 身份追踪探针
- [ ] 物体坐标只读探针（§5 风险 4）
- [ ] real 机械臂 PnP 外参标定（可选子项目，暂不启动）
- [ ] 0821 批（11 集）处置（沿用 sawvla 遗留，未裁决）
