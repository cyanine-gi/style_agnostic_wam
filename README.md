# style_agnostic_wam

风格无关（style-agnostic）世界模型 demo：以深度为隔离层，让世界模型学到与画面来源（仿真/真实）无关的表征。技术路线与阶段验收见 [docs/guideline.md](docs/guideline.md)。

## 环境配置

```bash
conda create -n style_agnostic_wam python=3.10 -y
conda activate style_agnostic_wam
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

- torch 2.9.1 的 PyPI linux wheel 自带 CUDA runtime，驱动 ≥ 550 即可，无需单独装 CUDA toolkit。
- 依赖版本全部 `==` 钉死，见 [requirements.txt](requirements.txt)。
- 目标硬件：单卡 16GB(开发基准 RTX 4070 Ti SUPER)。

## 模型准备

三个预训练模型统一经 ModelScope 下载到 `pretrained_models/`，运行时全程离线：

```bash
python scripts/download_pretrained.py --verify   # 全部下载并做加载自检
```

| 角色 | 模型 | 用途 |
|---|---|---|
| teacher | Depth Anything V2-L | 稠密深度教师，离线推理（§3.1） |
| encoder | DINOv2-S/14-reg | RGB 编码器，全程可微调（§2.3-E，v2 硬规格） |
| vlm | Qwen3-VL-2B-Instruct | Stage 3 VLA backbone(§3.6） |

模型路径与冻结边界写在 [configs/model.yaml](configs/model.yaml)，改模型只改这里。

## 数据准备

默认数据集（2026-09-11 起，Tienkung 已归档）：RoboMIND2.0 Franka 双臂，
real = `data/RoboMIND2.0-Franka-Part-1`，sim = `data/RoboMIND2.0-Franka-sim`
（ModelScope `X-Humanoid/RoboMIND2.0-Franka-Part-1` /
`X-Humanoid/RoboMIND2.0-Franka-sim`），matched 任务 hang_cup_on_cup_holder。
数据加载约定见 [docs/dataloader.md](docs/dataloader.md) 与
[configs/data.yaml](configs/data.yaml)。

数据检查/可视化（RGB/深度网格 + 动作曲线 + 报告）：

```bash
python scripts/check_franka_dataset.py    # 默认取两边文件名最小的 episode
```

## 数据可视化工具

下载完成后可以使用python scripts/visualize_robomind_hdf5.py [your hdf5 file path] 来查看数据包.
按Up Down Arrow 跳10帧,按Left Right Arrow跳1帧.

## 冒烟测试

数据与模型就绪后，用统一脚本验证三个模型都能在真实数据帧上推理：

```bash
python scripts/test_pretrained_models.py                 # 默认取 Franka real 一帧
python scripts/test_pretrained_models.py --hdf5 <path> --frame 500
```

输出包括：教师稠密深度与 GT 的对比图（存 `outputs/introspect/`，GT 洞标红）、教师 batch8@518 吞吐与全量推理工时估算、编码器 patch token 形状检查、VLM 图像描述回复及各模型峰值显存。

交互式查看单条 HDF5 轨迹（RGB-D 视频 + 状态曲线）：

```bash
python scripts/visualize_robomind_hdf5.py <trajectory.hdf5> --save
python scripts/visualize_robomind_hdf5.py <trajectory.hdf5> --summary-only   # 只打印结构摘要
```

## 关键设计裁决记录

> 只收录跨模块的口径性裁决；细节讨论记录在各设计文档与代码模块头。

- **Cycle 可逆性检验 = 运行时探针，不进训练损失**（2026-09-11 裁决）：
  前向动作推 k 步 → 输入完全相反的动作再推 k 步（动作空间是绝对关节
  位置，反向序列 = 原轨迹倒放，零构造代价）→ 与起始帧 latent 比较。
  它只用于运行时诊断（残差空间分布定位不可逆区域/接触事件、衡量 T 的
  累计展开误差），**禁止**作为一致性损失参与训练——机械臂可逆但场景
  不可逆（杯子被推走不会回来），全图一致性损失会教出"松手杯子弹回"
  的幻觉动力学。


