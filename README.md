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
| encoder | DINOv2-B/14 | Stage 0 冻结 RGB 编码器（§3.2） |
| vlm | Qwen3-VL-2B-Instruct | Stage 3 VLA backbone(§3.6） |

模型路径与冻结边界写在 [configs/model.yaml](configs/model.yaml)，改模型只改这里。

## 数据准备

git clone https://www.modelscope.cn/datasets/Dexmal/robotwin2-full.git

modelscope download  --dataset X-Humanoid/RoboMIND2.0-Tienkung --include data/tienkung/put_egg_into_box/* --local_dir ./RoboMIND2.0-Tienkung

git clone https://www.modelscope.cn/datasets/X-Humanoid/RoboMIND2.0-Tienkung-sim.git RoboMIND2.0-Tienkung-sim

## 数据可视化工具

下载完成后可以使用python scripts/visualize_robomind_hdf5.py [your hdf5 file path] 来查看数据包.
按Up Down Arrow 跳10帧,按Left Right Arrow跳1帧.

## 冒烟测试

数据与模型就绪后，用统一脚本验证三个模型都能在真实数据帧上推理：

```bash
python scripts/test_pretrained_models.py                 # 默认取 tidy_desktop 一帧
python scripts/test_pretrained_models.py --hdf5 <path> --frame 500
```

输出包括：教师稠密深度与 GT 的对比图（存 `outputs/introspect/`，GT 洞标红）、教师 batch8@518 吞吐与全量推理工时估算、编码器 patch token 形状检查、VLM 图像描述回复及各模型峰值显存。

交互式查看单条 HDF5 轨迹（RGB-D 视频 + 状态曲线）：

```bash
python scripts/visualize_robomind_hdf5.py <trajectory.hdf5> --save
python scripts/visualize_robomind_hdf5.py <trajectory.hdf5> --summary-only   # 只打印结构摘要
```


