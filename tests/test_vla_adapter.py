"""VLA latent adapter 单测：输出规格、space-to-depth 顺序约定、局部性、梯度。"""

import torch

from sawvla.models import VLALatentAdapter
from sawvla.models.vla_adapter import space_to_depth


def test_output_spec():
    adp = VLALatentAdapter()
    out = adp(torch.randn(2, 256, 384))
    assert out.shape == (2, 64, 2048)          # 8×8 网格 × Qwen3-VL-2B hidden


def test_space_to_depth_order_row_major():
    # 块内 row-major [(0,0),(0,1),(1,0),(1,1)]——MRoPE 坐标生成必须遵守同一约定
    x = torch.arange(4 * 4 * 1).reshape(1, 4, 4, 1).float()
    out = space_to_depth(x, s=2)               # (1, 2, 2, 4)
    # 输出 (0,0) 块应依次含输入 (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    assert out[0, 0, 0].tolist() == [0.0, 1.0, 4.0, 5.0]
    # 输出 (0,1) 块应依次含输入 (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    assert out[0, 0, 1].tolist() == [2.0, 3.0, 6.0, 7.0]


def test_output_token_locality():
    # 输出 token (i,j) 只能依赖输入 patch (2i:2i+2, 2j:2j+2)——绝对几何信息绑定
    adp = VLALatentAdapter()
    z = torch.zeros(1, 256, 384)
    out0 = adp(z)
    z2 = z.clone()
    z2[0, 3 * 16 + 5] = 100.0                  # 扰动 patch (3,5) → 应只影响输出 (1,2)
    out1 = adp(z2)
    changed = (out1 - out0).abs().sum(-1)[0]   # (64,)
    changed_idx = changed.nonzero().flatten().tolist()
    assert changed_idx == [1 * 8 + 2]


def test_gradient_flows():
    adp = VLALatentAdapter()
    z = torch.randn(1, 256, 384, requires_grad=True)
    adp(z).sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_param_count_small():
    adp = VLALatentAdapter()
    n = sum(p.numel() for p in adp.parameters())
    assert n < 1_000_000                       # 小头定位：384×512+LN ≈ 20 万


def test_rejects_wrong_token_count():
    adp = VLALatentAdapter()
    try:
        adp(torch.randn(1, 100, 384))
        raise AssertionError("应拒绝非 grid² 的 token 数")
    except ValueError:
        pass
