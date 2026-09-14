# style_agnostic_wam

## 概述

这是一个开源风格无关(style-agnostic)世界模型 repo,目标是:以深度,可逆性和在环强化目标为监督,全自动训练一个让世界模型学到与画面来源(仿真/真实)无关的表征,并能用于执行真实机器人的任务. 这三种监督信号可靠而廉价,可以用开源数据集和仿真环境大规模获取.

Photo-Realistic仿真往往是困难而不必要的,因为执行任务所必须的信息不在 Photo-Realistic 图像的画风中提供.这部分的特性需要由真机/仿真数字孪生进行一一对应的任务数据采集.为此,我们使用一些开源的数据集并进行对应的域对抗,以确保最后的世界模型中有效信息不来自 Photo-Realistic图像的风格,从而可以在大规模仿真数据上训练 VLA/WAM模型并确保性能不受损.

技术路线与阶段验收见 [docs/guideline.md](docs/guideline.md).

## 环境配置

```bash
conda create -n style_agnostic_wam python=3.10 -y
conda activate style_agnostic_wam
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

- torch 2.9.1 的 PyPI linux wheel 自带 CUDA runtime,驱动 ≥ 550 即可,无需单独装 CUDA toolkit.
- 依赖版本全部 `==` 钉死,见 [requirements.txt](requirements.txt).
- 目标硬件:单卡 16GB(开发基准 RTX 4070 Ti SUPER).

## 模型准备

三个预训练模型统一经 ModelScope 下载到 `pretrained_models/`,运行时全程离线:

```bash
python scripts/download_pretrained.py --verify   # 全部下载并做加载自检
```

| 角色 | 模型 | 用途 |
|---|---|---|
| teacher | Depth Anything V2-L | 稠密深度教师,离线推理(§3.1) |
| encoder | DINOv2-S/14-reg | RGB 编码器,全程可微调(§2.3-E,v2 硬规格) |
| vlm | Qwen3-VL-2B-Instruct | Stage 3 VLA backbone(§3.6) |

模型路径与冻结边界写在 [configs/model.yaml](configs/model.yaml),要更换模型,可以在这里修改配置.

## 数据准备

默认数据集:RoboMIND2.0 Franka 双臂,
real = `data/RoboMIND2.0-Franka-Part-1`,sim = `data/RoboMIND2.0-Franka-sim`
(ModelScope `X-Humanoid/RoboMIND2.0-Franka-Part-1` /
`X-Humanoid/RoboMIND2.0-Franka-sim`),matched 任务 hang_cup_on_cup_holder.
数据加载约定见 [docs/dataloader.md](docs/dataloader.md) 与
[configs/data.yaml](configs/data.yaml).

之前笔者调研过RoboMIND Tienkung 和 RoboMIND Tienkung-sim,可惜他们一个是夹爪一个是灵巧手,不能适配域无关的训练.

Franka双臂的仿真和真机数据在机位,布置上**也有一定的差别**(见 outputs/check_camera_views),因此笔者搭建了仿真环境,见"使用仿真环境生成新数据"一节.仿真效果见 outputs/sim_play目录.

*数据检查/可视化(RGB/深度网格 + 动作曲线 + 报告)*:

```bash
python scripts/check_franka_dataset.py    # 默认取两边文件名最小的 episode
```

## 数据可视化工具

下载完成后可以使用

```
python scripts/visualize_robomind_hdf5.py [your_hdf5_file_path] 
```

来查看数据包.

按 Up Down Arrow跳10帧,按 Left Right Arrow跳1帧.

## 冒烟测试

数据与模型就绪后,用统一脚本验证三个模型都能在真实数据帧上推理:

```bash
python scripts/test_pretrained_models.py # 默认取 Franka real 一帧
python scripts/test_pretrained_models.py --hdf5 <path> --frame 500
```

输出包括:教师稠密深度与 GT 的对比图(存 `outputs/introspect/`,GT 洞标红)、教师 batch8@518吞吐与全量推理工时估算、编码器 patch token形状检查、VLM 图像描述回复及各模型峰值显存.

交互式查看单条 HDF5 轨迹(RGB-D 视频 + 状态曲线):

```bash
python scripts/visualize_robomind_hdf5.py <trajectory.hdf5> --save
python scripts/visualize_robomind_hdf5.py <trajectory.hdf5> --summary-only   # 只打印结构摘要
```

## 使用仿真环境生成新数据

可以参考 src/simulation目录的 README.md配置本地仿真环境.


