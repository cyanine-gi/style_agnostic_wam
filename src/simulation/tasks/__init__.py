"""任务注册入口。注意：import 本包会加载 isaaclab 模块——
必须先进 SimContext 再 import（见 simulation/sim_context.py）。"""

import gymnasium as gym

gym.register(
    id="Saw-HangCup-FrankaDual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hang_cup.env_cfg:HangCupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.hang_cup.agents.rsl_rl_ppo_cfg:HangCupPPORunnerCfg",
    },
)


def default_env_cfg(task_id: str = "Saw-HangCup-FrankaDual-v0"):
    """按 task_id 返回默认 EnvCfg。新增任务时在此扩充分支。"""
    if task_id == "Saw-HangCup-FrankaDual-v0":
        from .hang_cup.env_cfg import HangCupEnvCfg
        return HangCupEnvCfg()
    raise KeyError(f"未知任务 {task_id}，请在 simulation/tasks/__init__.py 中注册并补充默认 cfg")
