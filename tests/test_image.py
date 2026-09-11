"""LetterboxPreprocessor（RGB 方案 B'）单元测试。

钉死填黑几何与两域同构性——这是"预处理不引入域标签"裁决（2026-09-11
二次裁决，见 src/sawvla/data/image.py 讨论记录）的可执行约束。
"""

import numpy as np
import torch

from sawvla.data import LetterboxPreprocessor
from sawvla.data.image import IMAGENET_MEAN, IMAGENET_STD


def test_letterbox_bounds_16_9_patch_aligned():
    """Franka 两域俯视相机 1280×720 (16:9)：内容 9 行 patch，上 3 下 4。"""
    pre = LetterboxPreprocessor(image_size=224, patch=14)
    ch, y0, y1, x0, x1 = pre.letterbox_bounds(720, 1280)
    assert ch == 9 * 14 == 126
    assert y0 == 3 * 14 == 42 and y1 == 42 + 126
    assert (x0, x1) == (0, 224)
    assert (224 - ch) % 14 == 0                     # 黑边恰好整数行 patch


def test_letterbox_bounds_4_3_and_square():
    pre = LetterboxPreprocessor(image_size=224, patch=14)
    # 4:3（腕部相机）：内容 12 行 patch，上下各 2 行
    ch, y0, y1, _, _ = pre.letterbox_bounds(480, 640)
    assert ch == 12 * 14 == 168 and y0 == 28
    # 方形：不填黑
    ch, y0, y1, _, _ = pre.letterbox_bounds(480, 480)
    assert ch == 224 and y0 == 0


def test_image_size_must_be_patch_multiple():
    with np.testing.assert_raises(ValueError):
        LetterboxPreprocessor(image_size=32, patch=14)


def test_output_spec_black_bars_and_normalization():
    pre = LetterboxPreprocessor(image_size=224, patch=14)
    img = np.zeros((720, 1280, 3), np.uint8)
    img[..., 0] = 255                                     # 纯红
    out = pre(img)
    assert out.shape == (3, 224, 224) and out.dtype == torch.float32
    # 内容区 R 通道 = (1-mean)/std
    np.testing.assert_allclose(out[0, 42:168, :].numpy(),
                               (1 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], rtol=1e-5)
    # 黑边区 = (0-mean)/std（上下黑带，不是 0——归一化在填黑之后）
    np.testing.assert_allclose(out[0, :42, :].numpy(),
                               (0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], rtol=1e-5)
    np.testing.assert_allclose(out[0, 168:, :].numpy(),
                               (0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], rtol=1e-5)


def test_domain_symmetric_treatment():
    """裁决核心：两域同为 16:9 ⇒ 同一内容经预处理后逐位一致、黑边一致。"""
    pre = LetterboxPreprocessor(image_size=224, patch=14)
    scene = np.random.default_rng(0).integers(0, 255, (720, 1280, 3),
                                              dtype=np.uint8)
    out_a, out_b = pre(scene), pre(scene.copy())
    assert torch.equal(out_a, out_b)
    # 黑边占比 = 7/16 行，恒定，与域无关
    black_rows = 3 * 14 + 4 * 14
    assert black_rows / 224 == 7 / 16


def test_full_fov_preserved():
    """方案 B' 相对裁剪方案的关键性质：画面左右边缘内容不丢。"""
    pre = LetterboxPreprocessor(image_size=224, patch=14)
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:, :40] = 255            # 左边缘亮条（模拟贴边的杯架）
    img[:, -40:] = 255           # 右边缘亮条
    out = pre(img)
    content = out[0, 42:168, :]          # 内容区 R 通道
    white = (1 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert content[:, 0].mean().item() > white * 0.9    # 左边缘仍在
    assert content[:, -1].mean().item() > white * 0.9   # 右边缘仍在


def test_dataset_uses_preprocessor(tmp_path):
    """dataset 端到端：16:9 样本经方案 B' 后是 (3, S, S)，上下黑带、内容居中。"""
    import cv2
    import h5py

    from test_dataset import CURVES, _vlen_write

    ep_dir = tmp_path / "data/franka/task/success_episodes/ep/data"
    ep_dir.mkdir(parents=True)
    rgbs, depths = [], []
    for _ in range(2):
        img = np.zeros((720, 1280, 3), np.uint8)
        img[:, :] = 200
        rgbs.append(cv2.imencode(".jpg", img)[1].tobytes())
        depths.append(cv2.imencode(".png", np.full((4, 4), 900, np.uint16))[1].tobytes())
    with h5py.File(ep_dir / "t.hdf5", "w") as f:
        _vlen_write(f, f"camera_observations/color_images/camera_front", rgbs)
        _vlen_write(f, f"camera_observations/depth_images/camera_front", depths)
        f.create_dataset("camera_color_channel/camera_front",
                         data=np.bytes_("rgb"))
        for k in CURVES:
            f.require_group(k).create_dataset("data", data=np.zeros((2, 8), np.float32))

    from sawvla.data import RoboMindDataset
    ds = RoboMindDataset(tmp_path, domain_id=0, camera="camera_front",
                         image_size=28)
    s = ds[0]
    assert s["rgb"].shape == (3, 28, 28)
    # 28px 网格：内容高 = round(28*720/1280/14)*14 = 14，上黑边 7px 不足一行
    # patch ⇒ 对齐到 0，内容为 [0:14]，下黑带 [14:28]
    bottom = s["rgb"][:, 14:, :]
    mid = s["rgb"][:, :14, :]
    black_r = (0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    np.testing.assert_allclose(bottom[0].numpy(), black_r, rtol=1e-4)
    assert mid.mean().item() > bottom.mean().item() + 1.0
    ds.close()
