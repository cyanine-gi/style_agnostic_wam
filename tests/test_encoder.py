"""编码器 E 单测：token 布局、可微调性、隐空间规格。

用本地真实权重（88MB，CPU 可跑）；权重缺失时跳过。
"""

import pytest
import torch

from conftest import ENCODER_PATH

pytestmark = pytest.mark.skipif(not ENCODER_PATH.is_dir(),
                                reason="编码器权重未下载")

from sawvla.models import DINOv2Encoder  # noqa: E402


@pytest.fixture(scope="module")
def encoder():
    return DINOv2Encoder(ENCODER_PATH)


def test_output_spec(encoder):
    z = encoder(torch.randn(2, 3, 224, 224))
    assert z.shape == (2, 256, 384)          # 16×16 网格 × d=384，无 CLS/register


def test_pos_encoding_interpolates(encoder):
    # 权重预训练分辨率 518，输入 224 必须能插值而不报错
    z = encoder(torch.randn(1, 3, 224, 224))
    assert torch.isfinite(z).all()


def test_trainable_by_default(encoder):
    # v2 硬规格：全程可微调，禁止冻结
    assert any(p.requires_grad for p in encoder.parameters())
    z = encoder(torch.randn(1, 3, 224, 224))
    z.sum().backward()
    grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_freeze_unfreeze(encoder):
    encoder.freeze()
    assert not any(p.requires_grad for p in encoder.parameters())
    encoder.unfreeze()
    assert all(p.requires_grad for p in encoder.parameters())


def test_param_count(encoder):
    n = sum(p.numel() for p in encoder.parameters()) / 1e6
    assert 20 < n < 25                        # DINOv2-S ≈ 22M
