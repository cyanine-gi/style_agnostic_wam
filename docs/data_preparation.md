# data_preparation.md — 数据准备（v2）

> 返回总览：[summary.md](summary.md)。存储根目录：**工程内 `data/`**（所有数据集、权重、token 缓存都放这里；`data/` 不进 git）。
> v2：场景定为 Franka Panda 桌面操作（方案 A）；数据全部优先取 LeRobot 现成格式，免自写 adapter；自建 IsaacLab 生成降级为可选补充。

## 1. 数据来源总表

| 来源 | 域 | 规模（本项目取用） | 体积（估） | 用途 |
|---|---|---|---|---|
| `nvidia/PhysicalAI-Robotics-Manipulation-SingleArm` | 仿真 | 全量 6 子集（~38k 条：开抽屉/开柜门/取放/堆叠等） | ~80GB | 仿真主训练集（带动作标签） |
| DROID LeRobot 移植（`cadene/droid_1.0.1` 或 `IPEC-COMMUNITY/droid_lerobot`） | 真机 | 同任务子集：按语言指令关键词过滤，目标 3–5k 条 | ~60–100GB | 真机主训练集 |
| `nvidia/PhysicalAI-Robotics-Manipulation-Augmented` | 仿真+照片增强 | 全量 1000 条 × 2 风格 | ~10GB | S2 风格不变性热身验证（同内容配对） |
| IsaacLab 自生成（可选） | 仿真 | 视 S2/S3 缺口再定 | 按需 | 补充视角/光照多样性（Isaac Lab Mimic 或 CuRobo 规划） |

预算合计 ~150–190GB。DROID 过滤后若不足 3k 条，放宽任务族关键词或改为全 DROID 随机子采样（世界模型自监督不依赖任务标签；任务配对只对 S2 验收必要）。

## 2. 下载

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 网络有问题时

python src/sawvla/data/download_physicalai.py --out data/physicalai_singlearm
python src/sawvla/data/download_physicalai.py --augmented --out data/physicalai_augmented
python src/sawvla/data/download_droid.py --filter-tasks configs/droid_task_keywords.yaml --target 5000 --out data/droid_subset
```

- DROID 过滤：`configs/droid_task_keywords.yaml` 维护任务关键词表（drawer/cabinet/pick/place/stack 等），按 episode 级语言指令匹配；过滤结果存 `droid_subset/matching_report.json`，抽 20 条人工核对"画面-指令-任务族"对齐。
- 网络兜底：`HF_ENDPOINT=https://hf-mirror.com`（本机直连 HF 不通，脚本已默认走镜像）；ModelScope 有 PhysicalAI 镜像（`nv-community/...`）。

### 实测数据（2026-09-07，脚本 dry-run + 小样本下载）

- DROID（`cadene/droid_1.0.1`，LeRobot v2.1）：95,600 episodes / 95 chunks / 15fps；关键词命中 **47,841** 条，其中 pick_place 45,405（关键词偏宽，" in the "/"put " 命中过泛）、drawer 4,032、cabinet 2,872、stack 1,243。PhysicalAI 侧只有 drawer/cabinet/stack 三类对齐，**建议收紧 pick_place 关键词或干脆只保 drawer/cabinet/stack 三族**（与仿真侧一一对应）。
- DROID 单 episode（exterior_1_left 一路视频 + parquet）≈ 4.5MB，5,000 条 ≈ 22GB，原 60–100GB 预算偏保守，可放宽 target 或加一路相机。
- PhysicalAI-SingleArm：6 个子数据集各为独立 LeRobot（30fps，world_camera + hand_camera + 两路 depth）；panda-open-drawer 有 1,273 episodes，单条约 0.3MB/相机，总体积将远小于 80GB 预估。
- 工程约束：本机直连 HF 不通；**文件列举（小流量）走官方 API + 本地代理 (见local_network.md)**（hf-mirror 分页 Link 头指回 huggingface.co，直连必挂），文件本体（大流量）走 hf-mirror —— 已在 `hf_utils.py` 实现，见 local_network.md。大仓库（DROID/SingleArm）枚举仍走 meta/info.json 模板，比逐页列举快几个数量级。

## 3. 统一格式（LeRobot v2，薄封装）

三源本身已是 LeRobot 格式，`to_lerobot.py` 只做统一化而非重写：

```
videos/{cam}.mp4    # 统一重编码 256×256, 10fps（WAM 训练用 16 帧片段）
{episode}.parquet   # observation.state, action(7-DoF 末端增量), task(语言), timestamp
meta/episodes.jsonl # episode_id, domain(sim|real|aug), source, embodiment, task_family, split
```

- **domain 字段只用于分析与消融采样，默认训练 pipeline 不把它喂给模型** —— 这是"风格无关"的实验纪律。
- 相机选择：统一用第三人称主视角（DROID: exterior cam；PhysicalAI: world cam），手腕相机留作扩展。
- 动作对齐：统一 7-DoF 末端增量 + 夹爪，`action_align.py`；DROID 无标签或低质量片段标 `action_valid=false`，只参与 latent action/WAM 自监督。
- 划分：sim 按场景、real 按环境 8:1:1；test 环境完全留出。
- **跨域配对集**（S2 验收用）：从 DROID 子集与 PhysicalAI 中各取同任务族片段（如"开抽屉"），按任务族+粗略动作方向配对，存 `pairs/s2_crossdomain_pairs.jsonl`，抽 30 对人工确认。

## 4. 混合采样规范

- `mix_sampler.py`：`--mix sim:real=1:1`（默认），按 batch 内比例采样；种子固定可复现。
- WAM 消融必须跑三种：`1:0`（仅仿真）、`0:1`（仅真机）、`1:1`（混合）。
- 每个 batch 记录来源统计到 `logs/mix_stats.jsonl`（事后核查比例漂移）。

## 5. 质量校验

- `validate.py`：视频可解码、帧数一致、动作维度=7、NaN 检查、相机键名映射（`observation.images.*`）齐全。
- 真机数据抽 20 条人工检查：画面-指令-动作三者对齐。
- 数据卡自动生成 `docs/datacard.md`（各源条数/时长/任务族分布/相机分布）。

## 6. 存储规划（工程内 data/）

| 目录 | 预算 |
|---|---|
| `data/physicalai_singlearm/` | 80GB |
| `data/droid_subset/` | 100GB |
| `data/physicalai_augmented/` | 10GB |
| `data/weights/` | 20GB |
| `data/token_cache/` | 50GB |
| `data/ckpt/` | 200GB |

`data/` 整目录进 `.gitignore`；版本号与各源 `meta/dataset_version.txt` 提交到 git（只提交元信息，不提交数据本体）。

## 7. 版本管理

- `meta/dataset_version.txt` 记录版本；config 引用版本号；大文件不进 git（DVC 或目录版本号）。
