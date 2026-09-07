# pretrained_model_dependency.md — 预训练权重与数据依赖（v2）

> 返回总览：[summary.md](summary.md)。权重放 `/data/sawvla/weights/`。下载失败依次尝试：`HF_ENDPOINT=https://hf-mirror.com` → ModelScope → gitee。
> v2：移除 SmolVLA/Octo/OpenVLA/π0/GR00T（VLA 线移出 v1）；新增数据集依赖与视频 tokenizer 选型。

## 1. 必需模型权重

| 模型 | 用途/阶段 | 权重位置 | 大小 | 显存 | 许可 |
|---|---|---|---|---|---|
| SigLIP-so400m | latent action 编码器初始化（S2） | `google/siglip-so400m-patch14-384` | ~800MB | <2GB | Apache 2.0 |
| Cosmos Tokenizer（DV 离散版） | WAM 帧 token 化（S3，仅推理） | `nvidia/Cosmos-Tokenizer-*` | ~1GB | <4GB | NVIDIA Open Model License（S0 核实条款） |
| Wan2.1 VAE（备选） | tokenizer 备选（S3） | 随 `Wan2.1` 发布 | ~1GB | <4GB | Apache 2.0 |
| CoTracker3 | 可选：latent action 可视化 | `facebook/cotracker3` | ~100MB | <4GB | CC BY-NC（仅研究，README 声明） |

## 2. 数据集依赖（本项目的一等公民）

| 数据集 | 域 | 位置 | 许可 |
|---|---|---|---|
| `nvidia/PhysicalAI-Robotics-Manipulation-SingleArm` | 仿真主训练集 | HF / ModelScope 镜像 `nv-community/...` | CC BY 4.0 |
| `nvidia/PhysicalAI-Robotics-Manipulation-Augmented` | 风格配对热身 | HF / ModelScope | CC BY 4.0 |
| DROID LeRobot 移植（`cadene/droid_1.0.1` / `IPEC-COMMUNITY/droid_lerobot`） | 真机主训练集 | HF | 原始 DROID 为 CC-BY 4.0，发布时以原始许可为准 |

## 3. 参考实现（读代码为主）

| 项目 | 仓库 | 借什么 |
|---|---|---|
| UniVLA | `github.com/OpenDriveLab/UniVLA` | latent action 学习范式、跨本体数据管线 |
| LAPA | `github.com/Latent-Action-Pretraining/LAPA` | VQ-VAE 帧对训练细节、无标注视频利用 |
| AdaWorld | arXiv 2503.18938 | 上下文不变 latent action 的自检方法 |
| WorldVLA / RynnVLA-002 | `github.com/alibaba-damo-academy` | 世界模型+VLA 统一架构 |
| V-JEPA 2-AC | `github.com/facebookresearch/vjepa2` | 潜空间动作条件预测（参考） |
| DreamGen / GR00T-Dreams | `github.com/NVIDIA/GR00T-Dreams` | 1:1 混合配方 |
| EnerVerse-AC | `github.com/AgibotTech/EnerVerse-AC` | 动作条件世界模型工程结构 |

## 4. 下载命令

```bash
pip install "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com   # 网络有问题时

huggingface-cli download google/siglip-so400m-patch14-384 --local-dir /data/sawvla/weights/siglip
# Cosmos Tokenizer 具体 repo id 在 S0 核实后写入
# 数据集下载走 src/sawvla/data/download_*.py，见 data_preparation.md
```

## 5. 显存红线（16GB）

| 任务 | 预算 |
|---|---|
| latent VQ-VAE（30M）全参 | ≤8GB（batch 128；爆则 64+累积2） |
| WAM（200M）全参 | ≤12GB（batch 32；爆则 16+累积2） |
| tokenizer 推理 | ≤4GB |
| 任何 7B 级模型微调 | ❌ 禁止 |

## 6. 许可红线

- 可商用：SigLIP、Wan2.1、PhysicalAI 数据（CC BY 4.0）、DROID（CC-BY 4.0）。
- 需 S0 核实：Cosmos Tokenizer 的 NVIDIA Open Model License 具体条款。
- 仅研究：CoTracker3（CC BY-NC）——README 必须声明；demo 视频素材只用自己的生成结果与 CC BY 数据，避免 NC 素材入镜。
