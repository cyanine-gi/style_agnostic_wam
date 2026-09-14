"""SimulationApp 生命周期封装：单例 + 上下文管理器。

动机（2026-09-13 裁决）：仿真环境与训练同进程，供单进程调试与
在线更新参数/在线学习。IsaacLab 的硬约束是 SimulationApp 必须先于一切
omni/isaaclab 模块导入启动、且每进程只能存在一个实例——本模块把这两条
约束集中到一个入口：

    from simulation.sim_context import SimContext

    with SimContext(headless=True) as ctx:
        env = ctx.make_env("Saw-HangCup-FrankaDual-v0", num_envs=1)
        ...

规则：
- **必须先进入 SimContext，再 import simulation.tasks**（任务包 import 时会
  加载 isaaclab 模块）；
- 一个进程内 SimContext 可多次进入/退出：首次进入启动 app，退出时只销毁
  env、app 保持存活（供训练进程反复激活仿真）；进程结束时 atexit 兜底关闭。
- 相机观测要求启动时 enable_cameras=True（默认开）。
"""

from __future__ import annotations

import atexit
from typing import Any


class SimContext:
    """SimulationApp 单例 + env 生命周期管理。"""

    _app = None          # 进程级单例 SimulationApp
    _launcher = None

    def __init__(self, headless: bool = True, device: str = "cuda:0",
                 enable_cameras: bool = True,
                 experience: str = "") -> None:
        self.headless = headless
        self.device = device
        self.enable_cameras = enable_cameras
        self.experience = experience
        self._envs: list[Any] = []

    # ------------------------------------------------------------------ #
    @classmethod
    def _ensure_app(cls, cfg: dict) -> None:
        if cls._app is not None:
            return
        # 注意：本 import 必须是进程内第一个 isaaclab 相关导入
        from isaaclab.app import AppLauncher
        cls._launcher = AppLauncher(cfg)
        cls._app = cls._launcher.app
        atexit.register(cls._shutdown)

    @classmethod
    def _shutdown(cls) -> None:
        if cls._app is not None:
            cls._app.close()
            cls._app = None
            cls._launcher = None

    # ------------------------------------------------------------------ #
    def __enter__(self) -> "SimContext":
        args = {"headless": self.headless, "device": self.device,
                "enable_cameras": self.enable_cameras}
        if self.experience:
            args["experience"] = self.experience
        self._ensure_app(args)
        return self

    def __exit__(self, *exc) -> None:
        self.close_envs()            # 只销毁 env，app 保持存活
        return False

    def make_env(self, task_id: str, num_envs: int | None = None,
                 cfg=None, render_mode: str | None = None):
        """创建 gym env（gym.make 路径，任务须已注册）。

        首次调用时才 import 任务包（触发 gym.register），保证 app 已启动。
        """
        assert SimContext._app is not None, "先进入 SimContext 上下文"
        import gymnasium as gym
        import simulation.tasks  # noqa: F401  # 注册全部任务

        if cfg is None:
            cfg = simulation.tasks.default_env_cfg(task_id)
            if num_envs is not None:
                cfg.scene.num_envs = num_envs
        env = gym.make(task_id, cfg=cfg, render_mode=render_mode)
        self._envs.append(env)
        return env

    def close_envs(self) -> None:
        for e in self._envs:
            try:
                e.close()
            except Exception:
                pass
        self._envs = []

    @property
    def app(self):
        return SimContext._app
