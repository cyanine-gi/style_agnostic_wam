# AGENTS.md — Style-Agnostic WAM

- 目标：真机+仿真视频混合训练、无需区分画面风格的世界-动作模型（WAM）demo，Franka Panda 桌面操作场景。GitHub 开源 + 录屏发布。
- 场景（方案 A）：仿真 = `nvidia/PhysicalAI-Robotics-Manipulation-SingleArm`（Isaac Sim 生成）；真机 = DROID LeRobot 移植的同任务子集；风格配对热身 = `nvidia/PhysicalAI-Robotics-Manipulation-Augmented`。
- 硬件：RTX 4070 Ti Super 16GB / 64GB RAM / 4TB / Ubuntu 24.04。
- 存储：所有数据集、模型权重、token 缓存、checkpoint 放工程内 `data/`（不进 git，只提交元信息）。
- 软件：IsaacLab v2.3.2 + Isaac Sim v5.1.0，已装在 conda 环境 `env_isaaclab`。
- 硬约束：PyTorch 2.x + CUDA 12.4，Python 3.11；单卡 16GB —— 只允许 ≤300M 自训模型全参训练；禁止微调任何 7B 级模型；WAM 帧 token 必须离线预计算。
- 栈：LeRobot 数据格式、Isaac Sim 5.x(pip) + IsaacLab（仅可选的数据补充生成用）、diffusers、transformers、Gradio。
- 范围纪律：v1 只做世界模型主线（数据 → latent action → WAM → 评测 → demo）。VLA/SmolVLA/LIBERO 是 future work，不提前写代码（见 architecture.md §8）。
- 入口：先读 `docs/summary.md`，它递归索引全部文档；阶段/脚本/验收以 `docs/stages.md` 为准。
- 规则：一个 stage 验收通过才进下一个；混合比例等关键超参必须可复现（写进 config）；不伪造实验数据；吞吐未实测前不写死训练工期；下载失败优先 hf-mirror / ModelScope / gitee。
- 尽一切可能用中转,如果小流量请求不通可以查看本地配置:local_network.md