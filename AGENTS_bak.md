# AGENTS.md — Style-Agnostic WAM/VLA

- 目标：真机+仿真视频混合训练、无需区分画面风格的 WAM（World-Action Model）+ VLA demo，机器人家务场景。
- 硬件：RTX 4070 Ti Super 16GB / 64GB RAM / 4TB / Ubuntu 24.04。
- 软件: IsaacLab v2.3.2; IsaacSim v5.1.0; 已安装在conda环境:env_isaaclab中.
- 硬约束：PyTorch 2.x + CUDA 12.4，Python 3.11；单卡 16GB —— VLA 只允许 SmolVLA/Octo 级微调；世界模型只允许 ≤1.3B LoRA 或 ≤300M 自训；禁止全参微调 OpenVLA/π0/GR00T。
- 栈：LeRobot 数据格式与训练栈、Isaac Sim 5.x(pip) + IsaacLab、diffusers、transformers、LIBERO 评测。
- 入口：先读 `docs/summary.md`，它递归索引全部文档；阶段/脚本/验收以 `docs/stages.md` 为准。
- 规则：一个 stage 验收通过才进下一个；混合比例等关键超参必须可复现（写进 config）；不伪造实验数据；下载失败优先 hf-mirror / ModelScope / gitee。
