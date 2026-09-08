# stages.md — 阶段规划（v2）

> 返回总览：[summary.md](summary.md)。每个阶段：目标 → 脚本入口 → 关键配置 → 验收标准。未验收不进下一阶段。
> v2：7 周四阶段；VLA/LIBERO 移出；新增 S2 两级自检与 S3 三组消融。

## S0 环境搭建（W1）

- 脚本：
  - `scripts/s0_setup_env.sh` —— 复用 conda `env_isaaclab`（py3.11, torch 2.x+cu124, Isaac Sim 5.1 + IsaacLab 2.3.2 已装）；补装 lerobot（源码）、diffusers/transformers、视频 tokenizer 依赖。
  - `src/sawvla/check_env.py` —— 自检：torch CUDA；**同进程 import Isaac Sim + lerobot**（依赖冲突尽早暴露）；视频 tokenizer 编码/解码一帧往返（打印峰值显存）；从 HF 拉一个 PhysicalAI 子集的 1 条 episode 并解码视频。
- 验收：全绿；tokenizer 往返峰值 < 4GB；**tokenizer 选型结论（Cosmos DV vs Wan2.1 VAE+RQ）写回 architecture.md §3**。
- 降配预案：tokenizer 显存超 → 降分辨率到 224 或逐帧编码。

## S1 统一数据层（W1–W2）

- 目标：三源统一为 LeRobot v2 薄封装 + 混合采样器；**元数据记录 domain 但训练默认不消费它**。
- 脚本：
  - `scripts/s1_download.sh` → `src/sawvla/data/download_{physicalai,droid}.py`
  - `scripts/s1c_to_lerobot.sh` → `src/sawvla/data/to_lerobot.py`（统一分辨率/帧率/相机键，非重写）
  - `src/sawvla/data/make_pairs.py` —— 生成 S2 验收用的跨域配对集 + Augmented 风格对清单
  - `src/sawvla/data/mix_sampler.py` —— 按比例混合采样（默认 sim:real=1:1，可配 1:0/0:1）
- 验收：`validate.py` 通过；混合采样器按种子可复现；跨域配对集 ≥ 200 对且抽检 30 对人工确认；数据卡 `docs/datacard.md` 自动生成。

## S2 Latent Action 表征（W2–W4）

- 目标：帧对 (o_t, o_{t+k}) → 离散 latent action 码，混合数据自监督训练，**不带域标签**。
- 脚本：
  - `scripts/s2_train_latent_action.sh` → `src/sawvla/train/train_latent_action.py`
  - 模型：`src/sawvla/latent/vqvae.py`（~30M；结构见 architecture.md）
- 配置：k∈{1,4} 两种步长；codebook 512×8 组；batch 128（爆显存降 64 + 累积 2）；bf16；lr 3e-4；20k step。**先实测吞吐再填工期预估**（v1 拍脑袋"1.5 天"不作数）。
- 验收（顺序执行，前不过后不测）：
  1. 重建 PSNR ≥ 28（held-out，k=1 版）；
  2. **热身——风格不变性**：Augmented 同内容双风格对的 latent action 码一致率显著高于随机配对；
  3. **主实验——跨域一致性**：跨域配对集最近邻匹配率显著高于随机基线（报告数值 + 置信区间）；
  4. 线性探测：latent action 预测真实 action（DROID 有标签）R² ≥ 0.5。

## S3 WAM 世界-动作模型（W4–W6）

- 目标：帧 token + latent action → 自回归预测未来帧 token；同一 latent action 序列可解码出双风格未来。
- 脚本：
  - `scripts/s3_tokenize.sh` —— 全量帧 token 离线预计算（存 `data/token_cache/`）
  - `scripts/s3_train_wam.sh` → `src/sawvla/train/train_wam.py --mix {1:0|0:1|1:1}`（**三组消融都要训**）
  - 模型：`src/sawvla/wam/ar_transformer.py`（~200M GPT 式；帧 token 见 architecture.md §3）
- 配置：序列 8 帧上下文 + 预测 8 帧；batch 32（爆显存降 16 + 累积 2）；lr 1e-4；60k step。
- 验收：
  1. held-out 未来帧预测 LPIPS 优于"复制末帧"基线 ≥ 40%（三组各报）；
  2. **卖点图产出**：≥ 10 组"同一 latent action → 仿真风格 + 真机风格"对照视频，人工检查动作语义一致；
  3. FVD 分域报告；
  4. **核心结论**：混合组在跨域指标（对另一域的 LPIPS/FVD）上优于单域组 —— 这是"混合 > 单域"的直接证据。

## S4 Demo + 发布（W6–W7）

- Demo：`src/sawvla/demo/app.py` —— Gradio 双栏：输入初始帧（sim/real 可选）+ latent action 序列（或从参考视频提取），输出双风格未来视频。
- 发布：
  - 自训权重（latent VQ-VAE + WAM ×3 消融组，总 < 8GB）传 HuggingFace；
  - README：卖点视频、核心结果表、许可声明（数据许可见 pretrained_model_dependency.md）；
  - `scripts/s4_reproduce.sh` 干净环境复现。
- 验收：核心结果表（下）+ 双风格生成视频 + README 完整。

## 关键结果表模板（最终交付）

| 训练数据 | 跨域匹配率↑ | 未来帧 LPIPS(sim)↓ | 未来帧 LPIPS(real)↓ | FVD(sim)↓ | FVD(real)↓ |
|---|---|---|---|---|---|
| WAM 仅仿真 | | | | | |
| WAM 仅真机 | | | | | |
| WAM 1:1 混合 | | | | | |

（S2 的 latent action 指标单独出表：风格不变性一致率 / 跨域匹配率 / 线性探测 R²，k=1 与 k=4 两行。）

## 工期备忘（v1 教训）

- 评测时间进工期：S3 三组 × 生成评测 ≥ 1 天机器时间。
- 吞吐未实测前不写死"X 天训完"。
- W2/W4/W6 的交叠是理想情况，单人串行时顺延，总预算 7 周 + 1 周缓冲。
