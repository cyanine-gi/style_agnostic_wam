"""世界模型网络模块（guideline v2 §2.3 / 隐空间与深度监督设计 §7/§8）。

- encoder:        DINOv2-S/14-registers 初始化，全程可微调，输出 16×16 patch token
- action_adapter: 动作块 + 本体感 → 条件 token（只进 T，不进 E）
- transition:     8 层 pre-LN Transformer，patch token 2D-RoPE，近恒等起步
- depth_decoder:  仅接 z 的卷积上采样头（禁止 encoder skip），输出 64×64 (μ, log σ)
- discriminator:  小容量域判别器，配合 losses.grl 使用
- vla_adapter:    【VLA 侧组件，非 WAM 本体】latent 256 token → 64 软 token 注入 LLM
"""

from .encoder import DINOv2Encoder
from .action_adapter import ActionAdapter
from .transition import TransitionModel
from .depth_decoder import DepthDecoder
from .discriminator import DomainDiscriminator
from .vla_adapter import VLALatentAdapter

__all__ = [
    "DINOv2Encoder",
    "ActionAdapter",
    "TransitionModel",
    "DepthDecoder",
    "DomainDiscriminator",
    "VLALatentAdapter",
]
