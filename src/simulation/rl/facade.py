"""RL 库包装层（2026-09-13 裁决：现在 rsl_rl，未来可换——
任务/脚本只调用本模块的接口，不直接 import 任何 RL 库）。

当前后端：rsl_rl（IsaacLab 默认）。换后端 = 在本模块加实现 +
改 DEFAULT_BACKEND，任务代码不动。
"""

from __future__ import annotations

DEFAULT_BACKEND = "rsl_rl"


def wrap_env_for_training(env):
    """gym env → 后端向量环境包装。"""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    return RslRlVecEnvWrapper(env)


def make_runner(env, agent_cfg, log_dir: str, device: str = "cuda:0"):
    """构建 on-policy runner（当前 = rsl_rl PPO）。"""
    from rsl_rl.runners import OnPolicyRunner
    return OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir,
                          device=device)


def load_policy(runner_log_dir: str, env, agent_cfg, device: str = "cuda:0"):
    """加载已训练策略（play/录制/可视化用）。runner_log_dir 可为含
    model.pt 的目录，也可直接给一个 .pt ckpt 文件路径。

    agent_cfg 必须与训练时同一个 runner cfg 类实例（如
    HangCupPPORunnerCfg()）——rsl_rl 3.x 的 OnPolicyRunner 构造时就要
    algorithm/policy/obs_groups 全量配置来建网络，再 load 权重；
    网络结构对不上 ckpt 会 load 失败。
    """
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    import os
    wrapped = env if isinstance(env, RslRlVecEnvWrapper) else RslRlVecEnvWrapper(env)
    resume_path = (runner_log_dir if str(runner_log_dir).endswith(".pt")
                   else os.path.join(runner_log_dir, "model.pt"))
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None,
                            device=device)
    runner.load(resume_path)
    return runner.get_inference_policy(device=device)
