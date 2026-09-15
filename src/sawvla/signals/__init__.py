"""监督信号注册表（设计动机与用法见 base.py 模块 docstring）。"""

from .base import (REGISTRY, SignalContext, SignalHead, SignalSet,
                   build_signals, register_signal)
from . import joint_pos  # noqa: F401  （导入即注册；新信号文件在此追加导入）

__all__ = ["REGISTRY", "SignalContext", "SignalHead", "SignalSet",
           "build_signals", "register_signal"]
