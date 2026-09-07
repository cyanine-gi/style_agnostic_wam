# architecture.md — 模块设计（v2）

> 返回总览：[summary.md](summary.md)。写/改模型与数据代码前必读。超参全部走 `configs/*.yaml`。
> v2：移除 VLA 模块（future work）；WAM 帧 token 改为离散 token + CE 损失（v1 的"连续 VAE latent + MSE"会塌缩成均值模糊帧，与像素级卖点图矛盾，已废弃）。

## 1. 模块依赖图

```
data/ ──> latent/vqvae ──(latent action 码)──> wam/ar_transformer ──> eval/, demo/
data/ ──> wam/tokenizer(冻结, 仅推理) ──(离散帧 token)──> wam/ar_transformer
```

## 2. Latent Action VQ-VAE（S2，~30M）

- 参考 LAPA/UniVLA：编码器输入帧对 `(o_t, o_{t+k})`（ViT-S 主干，可初始化为 SigLIP），输出动作隐向量 → 8 组 VQ（codebook 512）→ 离散 latent action 码 `z_a ∈ [8]`；解码器用 `z_a + o_t` 重建 `o_{t+k}`。
- 损失：重建 L2 + LPIPS + codebook commitment（β=0.25）；k∈{1,4} 双步长各训一版对比（WAM 交错序列用 k=1 那版）。
- **风格无关设计**：编码器输入做颜色抖动/风格增广；训练集混合 sim/real 且不给域标签；可选域对抗头（GRL，λ=0.1）作 ablation。
- 两级自检（验收用，见 stages.md）：
  1. **热身**：Augmented 数据集同内容双风格配对 → latent action 码应一致（风格不变性）；
  2. **主实验**：DROID↔PhysicalAI 跨域配对集 → latent action 码最近邻匹配率 vs 随机基线。

## 3. 帧 token 化（S3，冻结，仅推理）

- **主线**：NVIDIA Cosmos Tokenizer 离散视频 tokenizer（DV），帧 → 离散码。CE 损失的前提。
- **备选**：Wan2.1 VAE 连续 latent + 离线拟合的残差 VQ 量化层（若 Cosmos 许可/显存/效果有问题）。
- S0 必须实测确认选型并写回本节；所有帧 token 离线预计算存盘（`/data/sawvla/token_cache/`），**禁止训练时边编码边训**（吞吐打骨折）。

## 4. WAM 自回归世界-动作模型（S3，~200M）

- 序列：`[帧 token (8帧) + latent action token]` 交错排列，GPT 式因果 Transformer（width 1024, 16 层）。
- 训练目标：下一帧 token CE；条件 dropout 10%（支持无条件生成对照）。
- 生成：给定初始帧 + latent action 序列，自回归滚 8 帧，经 tokenizer 解码出像素；**双风格生成** = 同一 action 序列分别接仿真首帧与真机首帧各滚一次。
- 消融三个配置共用同一架构与超参：`sim-only` / `real-only` / `mix 1:1`。

## 5. 评测模块

- 表征级：
  - `eval/style_invariance.py` —— Augmented 同内容双风格对的 latent action 一致率；
  - `eval/crossdomain_match.py` —— 跨域配对集最近邻匹配率 vs 随机基线；
  - `eval/linear_probe.py` —— latent action 线性预测真实动作（DROID 有标签），R² 指标。
- 生成级：
  - `eval/lpips_future.py` —— held-out 未来帧预测 LPIPS，对照"复制末帧"基线；
  - `eval/fvd.py` —— 分域报告（仿真域/真机域分别算）。
- 评测脚本输出机器可读 json + 人工可读表格。

## 6. 关键配置文件

```
configs/data_mix_v1.yaml       # 三源路径、混合比例、版本号
configs/droid_task_keywords.yaml  # DROID 任务族过滤关键词
configs/latent_vqvae.yaml      # codebook 512×8, k=1/4, lr 3e-4
configs/wam_200m.yaml          # 8+8 帧, width1024, 16层
configs/ablation.yaml          # 1:0 / 0:1 / 1:1 矩阵
configs/eval.yaml
```

## 7. 编码约定

- PyTorch 2.x，bf16；LeRobot 数据集抽象优先，自写 Dataset 只做薄封装。
- 每次训练自动记录：数据版本、mix 比例、种子、git commit → `run_meta.json`。
- 单文件 ≤ 400 行；评测脚本输出机器可读 json + 人工可读表格。

## 8. Future work（v1 不做，勿提前写代码）

- WAM 表征迁移到 VLA：SmolVLA 微调 + latent action 辅助监督；评测走"混合预训练 → 目标域微调"标准协议（v1 废弃旧方案的原因：训练任务与 LIBERO 评测任务不重叠，zero-shot 评 LIBERO 的消融表无意义）。
- Wan2.1-1.3B LoRA 像素级渲染升级；AgiBot World + Genie Sim 第二场景。
