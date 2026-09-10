# 单元测试说明（tests/）

> 覆盖 [src/sawvla/models/](../src/sawvla/models/) 全部网络模块与
> [src/sawvla/losses/](../src/sawvla/losses/) 全部损失函数。
> 依据：guideline v2 §11.3（开发纪律）——每个 loss 配单测、GRL 符号方向必须验证。

## 一次跑完所有测试

```bash
# 在仓库根目录，使用项目 conda 环境
conda activate style_agnostic_wam
python -m pytest tests/ -q
```

- 预期结果：**53 passed**（测试数量随开发增长，以全绿为准）。
- 编码器相关测试需要本地权重 `pretrained_models/dinov2-small-reg`，缺失时自动 skip。
- 显存测试需要 CUDA GPU，无 GPU 环境自动 skip。
- 只看某个模块：`python -m pytest tests/test_losses.py -q`；
  跑单个测试：`python -m pytest tests/test_losses.py::test_metric_nll_empty_mask -q`。

---

## tests/test_encoder.py — 视觉编码器 E（DINOv2-S/14-reg）

| 测试 | 覆盖作用 |
|---|---|
| `test_output_spec` | 输出恰好 256×384（16×16 网格），CLS/register token 不进隐空间——隐空间是纯空间网格的硬规格 |
| `test_pos_encoding_interpolates` | 权重预训练分辨率 518、输入 224 时 2D 位置编码自动插值，uv 绑定不破坏 |
| `test_trainable_by_default` | 全程可微调（v2 硬规格，禁止冻结）；梯度确实流进 E |
| `test_freeze_unfreeze` | Stage 1 冻结 / 防漂移锚副本的 freeze/unfreeze 语义正确 |
| `test_param_count` | 参数量确为 DINOv2-S 量级（~22M），防误载 base/large 权重 |

## tests/test_transition.py — 转移模型 T + 动作适配器（逐步 token / block-causal）

| 测试 | 覆盖作用 |
|---|---|
| `test_adapter_per_step_tokens` | 逐步 token 化：k 步 = k 个 token；k 可变；第 j 个 token 只由第 j 步动作决定 |
| `test_forward_shape` | teacher-forced 多步前向：ẑ 与 z_seq 同形 (B, k, 256, 384) |
| `test_near_identity_at_init` | 零初始化残差头 ⇒ 初始 ẑ ≡ z（恒等起步，稳定早期训练） |
| `test_causal_leakage` | **因果泄漏（overall §3.4 硬性要求）**：只改未来动作 a₃，第 0/1/2 步预测逐位不变；第 3 步预测确实变（掩码未隔断动作通道的 sanity） |
| `test_action_changes_output` | 动作条件注入通路畅通：不同动作 → 不同预测 |
| `test_gradient_flows_to_input_and_cond` | 梯度同时回流到 z 与条件 token（打破零初始化后验证） |
| `test_rope_is_position_aware` | 2D-RoPE 生效：同一内容放不同网格位置，主干隐状态不同（uv 绑定） |
| `test_frame_embed_changes_output` | 帧索引 embedding 生效（多步展开的时间维编码） |
| `test_unroll_shape_and_matches_single_steps` | rollout 形状正确；回喂第 1 步与 teacher-forced 第 1 步一致 |
| `test_param_count` | 默认 8 层 ≈20M，远离 100M 上限 |

## tests/test_depth_decoder.py — 深度解码器 D

| 测试 | 覆盖作用 |
|---|---|
| `test_output_spec` | 输出 64×64 的 (μ, log σ) 双头 |
| `test_log_sigma_clamped` | log σ clamp 到 [-3, 5]，σ 不塌缩不爆炸（§11.3-2 要求） |
| `test_uv_channels_used` | 可选 uv 坐标通道确实接入入口（2 通道） |
| `test_gradient_flows_to_z` | 梯度只从 z 来（输入仅有 z，无 skip 捷径） |
| `test_param_count` | 轻量卷积头（≈1.7M），探针语义要求 D 保持弱而局部 |
| `test_rejects_wrong_token_count` | 非 grid² 的 token 数直接报错，防静默 reshape 错误 |

## tests/test_discriminator.py — 域判别器 + GRL

| 测试 | 覆盖作用 |
|---|---|
| `test_output_shape` | 每样本一个二分类 logit |
| `test_small_capacity` | 容量 < 2M（刻意小，防判别器过强导致 GRL 不稳） |
| `test_grl_negates_gradient` | **GRL 梯度符号确实反转**：经 GRL 的梯度 = −λ × 直接梯度（§11.3-3） |
| `test_grl_lambda_ramp_semantics` | λ=0 时上游收不到对抗梯度（ramp 起点的正确行为） |
| `test_grl_forward_is_identity` | GRL 前向是恒等（不扰动前向数值） |

## tests/test_losses.py — 深度损失 + 动力学损失

### L_metric（异方差 Laplacian NLL）

| 测试 | 覆盖作用 |
|---|---|
| `test_metric_nll_empty_mask` | 掩码全空 batch 返回 0、有梯度通路、不 NaN（§11.3-2） |
| `test_metric_nll_all_valid_and_prefers_accurate_mu` | 全有效 batch 正常；更准的 μ 损失更低（损失方向正确） |
| `test_metric_nll_ignores_masked_values` | 掩码外的极端值不影响结果（硬掩码，零梯度） |
| `test_metric_nll_nan_in_masked_region_safe` | 掩码外 NaN 不污染（real 域深度洞的常态） |

### L_teacher（scale-and-shift 对齐的教师蒸馏）

| 测试 | 覆盖作用 |
|---|---|
| `test_ssi_teacher_affine_invariant` | 教师相对深度乘加任意仿射变换后损失不变（相对深度的正确用法） |
| `test_ssi_teacher_zero_when_aligned` | μ 恰为教师仿射时损失 ≈ 0（实现无误） |
| `test_ssi_teacher_grad_flows` | 梯度正常流向 μ（对齐系数已 detach，无塌缩捷径） |

### L_grad（多尺度梯度）

| 测试 | 覆盖作用 |
|---|---|
| `test_multiscale_grad_empty_mask` | 掩码全空返回 0 且可反传 |
| `test_multiscale_grad_positive_on_wrong_prediction` | 平预测对非平目标损失 > 0 |
| `test_multiscale_grad_mask_aware_pooling` | **掩码感知降采样正确性**：无效像素不参与池化（§11.3-2） |
| `test_multiscale_grad_nan_masked_safe` | 掩码外 NaN 在多级池化后仍不污染 |

### L_smooth（RGB 边缘感知平滑）

| 测试 | 覆盖作用 |
|---|---|
| `test_edge_smooth_only_holes` | 仅空洞区（M=0）生效：改有效区 μ 损失不变 |
| `test_edge_smooth_flat_is_zero` | 平坦预测损失为 0 |
| `test_edge_smooth_rgb_downsampled_internally` | 允许传满分辨率 RGB，内部降采样 |

### 组合 DepthLoss 与 L_dyn

| 测试 | 覆盖作用 |
|---|---|
| `test_depth_loss_composition` | total = λ 加权和，权重与 §5.2 默认值一致；联合反传无 NaN |
| `test_depth_loss_empty_mask_still_finite` | 掩码全空时组合损失仍有限可反传 |
| `test_dyn_loss_basic_and_target_detached` | 目标一律 stop-gradient（梯度不进 z_target） |
| `test_dyn_loss_zero_when_perfect` | 完美预测损失为 0 |
| `test_dyn_loss_dynamic_weighting` | 动态区域加权生效：动态 patch 误差占比被放大 |

## tests/test_vla_adapter.py — VLA 侧 latent 注入 adapter

| 测试 | 覆盖作用 |
|---|---|
| `test_output_spec` | 256×384 → 64×2048（8×8 网格 × Qwen3-VL-2B hidden） |
| `test_space_to_depth_order_row_major` | 块内 row-major [(0,0),(0,1),(1,0),(1,1)] 顺序钉死——MRoPE position_ids 生成必须遵守同一约定 |
| `test_output_token_locality` | 输出 token (i,j) 只依赖输入 patch (2i:2i+2, 2j:2j+2)——绝对几何信息绑定不被破坏 |
| `test_gradient_flows` | 梯度可回流到 WAM latent（Stage 3 虽冻结 E，通路必须存在） |
| `test_param_count_small` | 小头定位：<1M 参数 |
| `test_rejects_wrong_token_count` | 非 grid² 输入直接报错 |

## tests/test_integration.py — 端到端冒烟
| 测试 | 覆盖作用 |
|---|---|
| `test_full_chain_one_step` | E → adapter → T → D + 判别器/GRL + 全部损失一次反传，**每个模块都收到非零梯度**——验证 §2 架构图的接口互相咬合 |

## tests/test_peak_memory.py — 最悲观显存（需 GPU）

| 测试 | 覆盖作用 |
|---|---|
| `test_peak_memory_all_modules_worst_case` | Stage 2 最悲观配置：全模块在卡（含防漂移锚副本）+ 4 步展开 + 每步解码与 GRL + AdamW 状态实际分配；断言峰值 < 13 GiB（§9 预算）。超预算即按降载顺序调整，测试失败 = 预警。当前实测 ≈4 GiB（batch 8, unroll 4），余量可供上调 batch/unroll |
