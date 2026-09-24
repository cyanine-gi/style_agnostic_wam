"""EpisodeWriter 落盘 schema 与 RoboMindDataset 读取的 round-trip 测试。

合成假 episode（按 discovery 布局落 tmp 目录）→ 用生产读取器
sawvla.data.dataset.RoboMindDataset 打开，验证：glob 命中、帧数、
rgb/depth/action/proprio 形状与数值、K/T 数据集 round-trip。
无 GPU/isaaclab 依赖（recording.py 顶层不 import isaaclab）。
"""

import h5py
import numpy as np
import pytest

from sawvla.data import RoboMindDataset
from simulation import recording as rec

CAMS = ["camera_front", "camera_left"]
T = 6


def _make_episode(root, ep_name="isaac-2026_09_17_00_00_00-0000",
                  task="hang_cup_on_rack", robot="franka_isaaclab"):
    """EpisodeWriter 落一集，返回 (h5_path, master, puppet, K, T)。"""
    rng = np.random.default_rng(0)
    writer = rec.EpisodeWriter(CAMS)
    master = rng.random((T, 16), dtype=np.float32)
    puppet = rng.random((T, 16), dtype=np.float32)
    for t in range(T):
        rgb = {c: np.full((16, 32, 3), (t * 20) % 255, np.uint8)
               for c in CAMS}
        depth = {c: np.full((12, 24), 800 + t, np.uint16) for c in CAMS}
        writer.add_frame(rgb, depth, master[t], puppet[t],
                         t * 1_000_000 // 15)
    K = {c: rec.pinhole_K(32, 16, 18.0 + i) for i, c in enumerate(CAMS)}
    Te = {c: rec.pose_to_matrix([i, -1.0, 1.5], (1, 0, 0, 0))
          for i, c in enumerate(CAMS)}
    b2r = {"left": np.eye(4), "right": np.eye(4) * 2}
    b2r["right"][3, 3] = 1.0
    h5 = (root / "data" / robot / task / "success_episodes" / ep_name
          / "data" / f"{ep_name}.hdf5")
    writer.save(h5, intrinsics=K, extrinsics=Te, base_to_robot=b2r)
    return h5, master, puppet, K, Te


def test_roundtrip_via_production_reader(tmp_path):
    h5, master, puppet, K, Te = _make_episode(tmp_path)
    ds = RoboMindDataset(tmp_path, domain_id=9, camera="camera_front")
    assert len(ds.episodes) == 1 and len(ds) == T
    item = ds[3]                                   # 第 4 帧
    assert item["rgb"].shape[0] == 3               # (3,S,S) letterbox
    assert item["action"].shape == (16,)
    assert item["proprio"].shape == (16,)
    # 动作拼接顺序 [L7,R7,gL,gR]：右臂 = 写入 master 的 7:14
    np.testing.assert_allclose(item["action"][7:14].numpy(),
                               master[3, 7:14], atol=1e-6)
    np.testing.assert_allclose(item["proprio"][7:14].numpy(),
                               puppet[3, 7:14], atol=1e-6)
    assert item["is_intervene"] is False
    ds.close()


def test_kt_roundtrip(tmp_path):
    h5, *_rest, K, Te = _make_episode(tmp_path)
    with h5py.File(h5, "r") as f:
        for c in CAMS:
            np.testing.assert_allclose(
                f[f"camera_intrinsics/{c}/matrix"][:], K[c], atol=1e-9)
            np.testing.assert_allclose(
                f[f"camera_extrinsics/{c}"][:], Te[c], atol=1e-9)
            assert f[f"camera_color_channel/{c}"][()] == b"rgb"
            assert tuple(f[f"camera_color_resolution/{c}"][:]) == (32, 16)
        assert f["metadata/trajectory_length"][()] == T
        assert "base_to_robot_transformation/base_to_robot_left" in f


def test_failed_episode_not_discovered(tmp_path):
    """failed_episodes 布局天然被 success_episodes glob 排除。"""
    h5, *_ = _make_episode(tmp_path)
    failed = str(h5).replace("success_episodes", "failed_episodes")
    import shutil
    from pathlib import Path
    Path(failed).parent.mkdir(parents=True)
    shutil.move(str(h5), failed)
    with pytest.raises(FileNotFoundError):
        RoboMindDataset(tmp_path, domain_id=9, camera="camera_front")


def test_empty_episode_rejected(tmp_path):
    writer = rec.EpisodeWriter(CAMS)
    with pytest.raises(ValueError):
        writer.save(tmp_path / "x.hdf5", intrinsics={}, extrinsics={},
                    base_to_robot={})
