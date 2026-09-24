"""recording.py 纯数学件单元测试（无 GPU/isaaclab 依赖）。

覆盖：pinhole_K 公式、四元数→旋转矩阵、位姿合成与外参基座变换、
夹爪线性翻转（连续性裁决）、相机位姿采样范围与可复现性。
"""

import numpy as np
import pytest

from simulation import recording as rec


# --------------------------------------------------------------------------- #
def test_pinhole_K_formula():
    K = rec.pinhole_K(1280, 720, 18.0)
    fx = 1280 * 18.0 / 20.955
    assert K[0, 0] == pytest.approx(fx)
    assert K[1, 1] == pytest.approx(fx)          # 方形像素 fx==fy
    assert K[0, 2] == 640.0 and K[1, 2] == 360.0
    assert K[2, 2] == 1.0


def test_quat_wxyz_to_matrix_known_rotations():
    np.testing.assert_allclose(
        rec.quat_wxyz_to_matrix((1, 0, 0, 0)), np.eye(3), atol=1e-12)
    # 绕 z 转 90°：x 轴 → y 轴
    R = rec.quat_wxyz_to_matrix((np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)))
    np.testing.assert_allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(R @ R @ R @ R, np.eye(3), atol=1e-12)


def test_extrinsics_in_base_hand_computed():
    """相机在世界系 (px,py,pz) 无旋转，基座中点 (bx,by,bz) ⇒
    T_base_cam 平移 = p − b，旋转 = I。"""
    T_world_base = np.eye(4)
    T_world_base[:3, 3] = [0.0, 0.15, 0.75]
    T_world_cam = rec.pose_to_matrix([0.0, -1.05, 1.55], (1, 0, 0, 0))
    T_base_cam = np.linalg.inv(T_world_base) @ T_world_cam
    np.testing.assert_allclose(T_base_cam[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T_base_cam[:3, 3], [0.0, -1.20, 0.80],
                               atol=1e-12)


def test_flip_gripper16_continuous():
    """线性翻转保连续（2026-09-17 裁决：禁止二值化）：端点对换、
    中点不动、臂关节不动。"""
    a = np.arange(16, dtype=np.float32) * 0.0
    a[14], a[15] = 0.0, 0.04
    out = rec.flip_gripper16(a)
    assert out[14] == pytest.approx(0.04, abs=1e-6)
    assert out[15] == pytest.approx(0.0, abs=1e-6)
    mid = rec.flip_gripper16(np.array([0.0] * 14 + [0.02, 0.017],
                                      dtype=np.float32))
    assert mid[14] == pytest.approx(0.02, abs=1e-6)
    assert mid[15] == pytest.approx(0.023, abs=1e-6)   # 连续，不是 0/1
    arm = np.linspace(-1, 1, 14, dtype=np.float32)
    full = np.concatenate([arm, [0.01, 0.03]]).astype(np.float32)
    np.testing.assert_allclose(rec.flip_gripper16(full)[:14], arm)


def test_sample_camera_poses_ranges_and_seed():
    cams = {
        "front": {"pos": [0.0, -1.05, 1.55], "look_at": [0.0, 0.0, 0.75],
                  "focal_length_mm": 18.0,
                  "randomize": {"pos_delta_m": [0.15, 0.15, 0.15],
                                "look_at_delta_m": [0.05, 0.05, 0.05],
                                "focal_range_mm": [16.0, 20.0]}},
        "fixed": {"pos": [1.0, 0.0, 1.0], "look_at": [0.0, 0.0, 0.75],
                  "focal_length_mm": 18.0},      # 无 randomize → 原值
    }
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rec.sample_camera_poses(cams, rng)
        d = np.abs(p["front"]["pos"] - np.array([0.0, -1.05, 1.55]))
        assert (d <= 0.15 + 1e-12).all()
        dl = np.abs(p["front"]["look_at"] - np.array([0.0, 0.0, 0.75]))
        assert (dl <= 0.05 + 1e-12).all()
        assert 16.0 <= p["front"]["focal_mm"] <= 20.0
        np.testing.assert_allclose(p["fixed"]["pos"], [1.0, 0.0, 1.0])
        assert p["fixed"]["focal_mm"] == 18.0
    # 同种子可复现
    a = rec.sample_camera_poses(cams, np.random.default_rng(7))["front"]
    b = rec.sample_camera_poses(cams, np.random.default_rng(7))["front"]
    np.testing.assert_allclose(a["pos"], b["pos"])
    assert a["focal_mm"] == b["focal_mm"]
