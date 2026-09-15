"""可插拔监督信号注册表（2026-09-14 用户裁决）。

动机（隐空间分区）：patch token 由深度损失显式塑形（局部信息）；
register 是全局槽位，由用户注册的**全局信号**隐式/自监督塑形——
例如关节位置回归（本体感）、未来"杯子到桌面高度"等任务量。

新增一个信号 = 三步，trainer 不动：
1. 写 SignalHead 子类（forward / target / loss）；
2. @register_signal("名字") 装饰；
3. configs/model.yaml 的 signals 节加一行（enabled / weight）。

挂接约定：
- 类属性 reads 指定从 SignalContext 的哪个通道读特征：
  "reg_agnostic"（默认，域无关 register——全局信号的规定住所）、
  "reg_domain" / "registers" / "patch" / "cls"；
- target(batch) 从 dataloader batch 取监督目标（张量）；
- loss(pred, target) 返回标量；权重由 model.yaml signals.<name>.weight 给，
  SignalSet.losses 返回的是**加权后**的损失（直接加进总损失）。

分区纪律（同日裁决）：全局信号默认只读 reg_agnostic；读 patch 会把局部
几何泄进全局通道，读 reg_domain 会被域信息污染——都需要显式理由。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SignalContext:
    """encoder.forward_tokens 的输出 + register 角色划分（trainer 构建）。"""

    cls: torch.Tensor               # (B, d)
    registers: torch.Tensor         # (B, n_reg, d)
    patch: torch.Tensor             # (B, N, d)
    roles: dict                     # {"domain": [int], "agnostic": [int]}

    def channel(self, name: str) -> torch.Tensor:
        if name == "reg_domain":
            return self.registers[:, self.roles["domain"]]
        if name == "reg_agnostic":
            return self.registers[:, self.roles["agnostic"]]
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(f"未知信号通道 {name!r}，可用：cls/registers/patch/"
                           f"reg_domain/reg_agnostic") from None


class SignalHead(nn.Module):
    """信号头基类。子类必须定义 reads、forward、target、loss。"""

    name: str                       # 由 register_signal 装饰器写入
    reads: str = "reg_agnostic"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def target(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


REGISTRY: dict[str, type[SignalHead]] = {}


def register_signal(name: str):
    def deco(cls: type[SignalHead]) -> type[SignalHead]:
        if name in REGISTRY:
            raise KeyError(f"信号 {name!r} 重复注册："
                           f"{REGISTRY[name].__name__} 与 {cls.__name__}")
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


class SignalSet(nn.Module):
    """按 model.yaml signals 节装配的一组信号头（空清单 = 空集合）。"""

    def __init__(self, model_cfg: dict, dim: int, n_reg: int) -> None:
        super().__init__()
        self.roles = model_cfg["encoder"]["register_roles"]
        n_tokens = {
            "cls": 1,
            "registers": n_reg,
            "reg_domain": len(self.roles["domain"]),
            "reg_agnostic": len(self.roles["agnostic"]),
            "patch": model_cfg["latent"]["grid_size"] ** 2,
        }
        heads, weights = [], {}
        for name, sc in (model_cfg.get("signals") or {}).items():
            if not sc.get("enabled", True):
                continue
            if name not in REGISTRY:
                raise KeyError(f"signals.{name} 未注册（已注册："
                               f"{sorted(REGISTRY)}）——先写 SignalHead 子类")
            cls = REGISTRY[name]
            heads.append(cls(dim=dim, n_tokens=n_tokens[cls.reads]))
            weights[name] = float(sc.get("weight", 1.0))
        self.heads = nn.ModuleList(heads)
        self.weights = weights

    def losses(self, ctx: SignalContext, batch: dict) -> dict[str, torch.Tensor]:
        """{信号名: 加权后标量损失}；目标自动搬到 pred 的设备/精度。"""
        out = {}
        for head in self.heads:
            pred = head(ctx.channel(head.reads))
            tgt = head.target(batch).to(device=pred.device, dtype=pred.dtype)
            out[head.name] = self.weights[head.name] * head.loss(pred, tgt)
        return out


def build_signals(model_cfg: dict, dim: int, n_reg: int) -> SignalSet:
    return SignalSet(model_cfg, dim=dim, n_reg=n_reg)
