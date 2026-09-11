"""Stage0MotionThinnedDataset 单元测试：按累计运动量抽稀的索引重映射。

用合成 HDF5（复用 test_dataset 的 episode 构造器）造已知运动模式：
前静后动（线性斜坡），钉死贪心保留规则（累计 ||Δq||₂ ≥ τ 保留、
首末帧恒保留）、索引映射正确性、以及 τ=0 时与基类逐帧一致。
不依赖真实数据。
"""

import numpy as np
import pytest

from sawvla.data import Stage0MotionThinnedDataset
from test_dataset import ARM_KEYS, EE_KEYS, make_episode

N = 20          # 帧数
RAMP_AT = 10    # 前 10 帧静止，之后线性斜坡


def make_motion_episode(tmp_path, step=0.02):
    """前 RAMP_AT 帧全零，之后每帧每个关节 +step（14 关节斜坡）。"""
    curves = {}
    ramp = np.zeros((N, 8), np.float32)
    ramp[RAMP_AT:] = np.arange(1, N - RAMP_AT + 1, dtype=np.float32)[:, None] * step
    for k in ARM_KEYS:
        curves[k] = ramp.copy()
    curves.update({k: np.zeros((N, 1), np.float32) for k in EE_KEYS})
    make_episode(tmp_path, "task_a", "ep_001", n_frames=N)
    # make_episode 写的是随机曲线，覆写为受控运动模式
    import h5py
    h5 = next(tmp_path.glob("data/*/*/success_episodes/*/data/*.hdf5"))
    with h5py.File(h5, "a") as f:
        for k, v in curves.items():
            f[f"{k}/data"][:] = v
    return curves


def expected_kept(step, tau, n=N, ramp_at=RAMP_AT):
    """与实现无关的参考解：纯 Python 重算贪心保留。"""
    d = np.zeros(n)
    d[ramp_at:] = step * np.sqrt(14)          # 每帧 14 关节各动 step
    kept, acc = [0], 0.0
    for i in range(1, n):
        acc += d[i]
        if acc >= tau:
            kept.append(i)
            acc = 0.0
    if kept[-1] != n - 1:
        kept.append(n - 1)
    return kept


def test_greedy_keep_matches_reference(tmp_path):
    make_motion_episode(tmp_path, step=0.02)
    tau = 0.1
    ds = Stage0MotionThinnedDataset(tmp_path, domain_id=0,
                                    camera="camera_front",
                                    motion_thresh=tau, verbose=False)
    exp = expected_kept(0.02, tau)            # 斜坡 Δ=0.02·√14≈0.0748
    assert exp == [0, 11, 13, 15, 17, 19]     # 静止段只剩 frame 0
    assert len(ds) == len(exp)
    for idx, frame in enumerate(exp):
        assert ds.locate(idx) == (0, frame)
    ds.close()


def test_motion_signal_excludes_gripper(tmp_path):
    """抽稀只看臂关节：EE 大幅跳变不影响保留结果。"""
    curves = make_motion_episode(tmp_path, step=0.02)
    import h5py
    h5 = next(tmp_path.glob("data/*/*/success_episodes/*/data/*.hdf5"))
    with h5py.File(h5, "a") as f:             # EE 全部置 1（巨大跳变）
        for k in EE_KEYS:
            f[f"{k}/data"][:] = 1.0
    ds = Stage0MotionThinnedDataset(tmp_path, domain_id=0,
                                    camera="camera_front",
                                    motion_thresh=0.1, verbose=False)
    assert len(ds) == 6                       # 与纯臂参考解一致
    ds.close()


def test_zero_thresh_is_identity(tmp_path):
    """τ=0 关闭抽稀：长度与映射与基类全帧一致。"""
    make_motion_episode(tmp_path)
    ds = Stage0MotionThinnedDataset(tmp_path, domain_id=0,
                                    camera="camera_front",
                                    motion_thresh=0.0, verbose=False)
    assert len(ds) == N
    assert [ds.locate(i) for i in range(N)] == [(0, i) for i in range(N)]
    ds.close()


def test_getitem_frame_id_consistency(tmp_path):
    """抽稀后 __getitem__ 的 frame_id 与曲线内容一致（取斜坡段验证）。"""
    curves = make_motion_episode(tmp_path, step=0.02)
    ds = Stage0MotionThinnedDataset(tmp_path, domain_id=0,
                                    camera="camera_front",
                                    motion_thresh=0.1, verbose=False)
    s = ds[2]                                  # exp[2] = 真实帧 13
    assert s["frame_id"] == 13
    expect = curves["master/arm_left_position_align"][13, :7]
    np.testing.assert_allclose(s["action"].numpy()[:7], expect, rtol=1e-6)
    ds.close()


def test_locate_out_of_range(tmp_path):
    make_motion_episode(tmp_path)
    ds = Stage0MotionThinnedDataset(tmp_path, domain_id=0,
                                    camera="camera_front",
                                    motion_thresh=0.1, verbose=False)
    with pytest.raises(IndexError):
        ds.locate(6)
    ds.close()
