"""RoboMindDataset / RawDepthSupervision / FrankaJointGripperPreprocessor 单元测试。

用合成 HDF5（JPEG color + PNG depth + 状态曲线，复刻 RoboMIND2.0-Franka
schema）验证：episode 发现、帧解码、通道序标记处理、唯一掩码规则
（含 65535 哨兵）、臂曲线裁 7 维 + EE 拼接顺序、标签字段。
不依赖真实数据（真实数据的冒烟验证由 scripts/check_franka_dataset.py 负责）。
"""

import cv2
import h5py
import numpy as np
import pytest
import torch

from sawvla.data import (FrankaJointGripperPreprocessor,
                         OnlineDisparitySupervision, RawDepthSupervision,
                         RoboMindDataset)

CAM = "camera_front"
OTHER_CAM = "camera_top"
ARM_KEYS = ["master/arm_left_position_align", "master/arm_right_position_align",
            "puppet/arm_left_position_align", "puppet/arm_right_position_align"]
EE_KEYS = ["master/end_effector_left_position_align",
           "master/end_effector_right_position_align",
           "puppet/end_effector_left_position_align",
           "puppet/end_effector_right_position_align"]
CURVES = ARM_KEYS + EE_KEYS


def _vlen_write(f, path, buffers):
    f.require_group(path.rsplit("/", 1)[0])
    ds = f.create_dataset(path, (len(buffers),),
                          dtype=h5py.vlen_dtype(np.dtype("uint8")))
    for i, b in enumerate(buffers):
        ds[i] = np.frombuffer(b, np.uint8)


def make_episode(path, task_dir, ep_name, n_frames=5, seed=0,
                 depth_kind="mixed", channel="rgb"):
    """在正确目录布局下造一个 episode，返回曲线字典。

    臂曲线造 8 维（复刻 real 的 7 关节 + 夹爪），钉死预处理器裁 [:7] 的
    行为；EE 曲线 1 维。
    """
    rng = np.random.default_rng(seed)
    ep_dir = path / "data" / "franka" / task_dir / "success_episodes" / ep_name / "data"
    ep_dir.mkdir(parents=True)
    h5 = ep_dir / "trajectory.hdf5"

    rgbs, depths = [], []
    for i in range(n_frames):
        img = np.full((8, 16, 3), (seed * 40 + i * 10) % 255, np.uint8)  # 纯色 JPEG 近似无损
        rgbs.append(cv2.imencode(".jpg", img)[1].tobytes())
        if depth_kind == "mixed":      # 0 洞 + 超量程 + 65535 哨兵 + 有效值
            d = np.full((6, 7), 1000 + i, np.uint16)
            d[0, 0] = 0
            d[1, 1] = 20000
            d[2, 2] = 65535
        else:                          # 全有效
            d = np.full((6, 7), 800, np.uint16)
        depths.append(cv2.imencode(".png", d)[1].tobytes())

    curves = {k: rng.random((n_frames, 8), dtype=np.float32) for k in ARM_KEYS}
    curves.update({k: rng.random((n_frames, 1), dtype=np.float32)
                   for k in EE_KEYS})

    with h5py.File(h5, "w") as f:
        _vlen_write(f, f"camera_observations/color_images/{CAM}", rgbs)
        _vlen_write(f, f"camera_observations/depth_images/{CAM}", depths)
        f.create_dataset("camera_observations/is_intervene",
                         data=np.arange(n_frames) % 2 == 0)
        f.create_dataset("camera_observations/timestamp",
                         data=np.arange(n_frames) * 33333)
        f.create_dataset(f"camera_color_channel/{CAM}",
                         data=np.bytes_(channel))
        for k, v in curves.items():
            g = f.require_group(k)
            g.create_dataset("data", data=v)
    return curves


@pytest.fixture()
def two_task_root(tmp_path):
    make_episode(tmp_path, "task_b", "ep_002", n_frames=4, seed=2)
    curves_a = make_episode(tmp_path, "task_a", "ep_001", n_frames=5, seed=1)
    return tmp_path, curves_a


def make_ds(root, **kw):
    kw.setdefault("camera", CAM)
    return RoboMindDataset(root, domain_id=0, **kw)


# --------------------------------------------------------------------------- #
def test_discovery_and_length(two_task_root):
    root, _ = two_task_root
    ds = make_ds(root)
    assert len(ds.episodes) == 2
    assert len(ds) == 4 + 5                     # 总帧数
    assert ds.tasks == ["task_a", "task_b"]     # 字典序
    ds.close()


def test_tasks_filter(two_task_root):
    root, _ = two_task_root
    ds = make_ds(root, tasks=["task_b"])
    assert len(ds.episodes) == 1 and len(ds) == 4
    ds.close()


def test_locate_mapping(two_task_root):
    root, _ = two_task_root
    ds = make_ds(root)
    # episode 排序：task_a/ep_001（5 帧）在前，task_b/ep_002（4 帧）在后
    assert ds.locate(0) == (0, 0)
    assert ds.locate(4) == (0, 4)
    assert ds.locate(5) == (1, 0)
    assert ds.locate(8) == (1, 3)
    with pytest.raises(IndexError):
        ds.locate(9)
    ds.close()


def test_sample_schema_and_labels(two_task_root):
    root, _ = two_task_root
    ds = make_ds(root, image_size=28)
    s = ds[0]
    assert s["rgb"].shape == (3, 28, 28) and s["rgb"].dtype == torch.float32
    assert s["depth"].shape == (64, 64) and s["depth"].dtype == torch.float32
    assert s["mask"].shape == (64, 64)
    assert s["action"].shape == (16,) and s["proprio"].shape == (16,)
    assert s["domain"] == 0 and s["task_id"] == 0     # task_a 字典序为 0
    assert s["episode_id"] == 0 and s["frame_id"] == 0
    assert s["is_intervene"] is True                  # frame 0: arange%2==0
    assert ds[1]["is_intervene"] is False
    ds.close()


def test_depth_raw_passthrough_and_unique_mask_rule(two_task_root):
    """RawDepthSupervision（显式注入）：原始分辨率直通 + 唯一掩码规则。"""
    root, _ = two_task_root
    ds = make_ds(root, supervision=RawDepthSupervision(1, 10000))
    s = ds[2]                                          # 深度应全为 1002
    interior = s["depth"][2, 3].item()
    assert interior == 1002.0                          # 原始值直通，无变换
    assert s["mask"][0, 0].item() == 0.0               # 0 = 洞
    assert s["mask"][1, 1].item() == 0.0               # 20000 超量程
    assert s["mask"][2, 2].item() == 0.0               # 65535 哨兵
    assert s["mask"][2, 3].item() == 1.0
    assert s["mask"].sum().item() == 6 * 7 - 3         # 仅三处无效
    ds.close()


def test_online_disparity_supervision_geometry():
    """在线 64×64 disparity：与方案 B' 逐 patch 对齐的填黑几何。"""
    sup = OnlineDisparitySupervision(1, 10000)
    ch, y0, y1 = sup.letterbox_bounds(720, 1280)
    assert ch == 36 and y0 == 12 and y1 == 48          # 16:9 → 内容 36 行、上 12
    # RGB 侧对照：LetterboxPreprocessor(224, patch14) 的 42:168 按 64/224
    # 缩放即 12:48——逐 patch 一致
    d = np.full((720, 1280), 500, np.uint16)           # 全图 500mm
    out = sup(d)
    assert out["depth"].shape == (64, 64)
    assert out["mask"][:12].sum() == 0 and out["mask"][48:].sum() == 0
    assert out["depth"][:12].max() == 0 and out["depth"][48:].max() == 0
    np.testing.assert_allclose(out["depth"][12:48], 1000.0 / 500, rtol=1e-5)
    assert out["mask"][12:48].min() == 1.0


def test_online_disparity_supervision_mask_rule():
    """洞/哨兵/超量程不参与均值；掩码按有效比例阈值重建。"""
    sup = OnlineDisparitySupervision(1, 10000)
    d = np.full((720, 1280), 1000, np.uint16)
    d[:, :640] = 0                                     # 左半全洞
    d[0, 641] = 65535                                  # 哨兵
    out = sup(d)
    assert out["mask"][12:48, :32].sum() == 0          # 左半无效
    assert out["mask"][12:48, 40:].min() == 1.0        # 右半有效
    np.testing.assert_allclose(out["depth"][12:48, 40:], 1.0, rtol=1e-5)


def test_action_proprio_concat_order_and_arm_slice(two_task_root):
    """action/proprio = [L臂7, R臂7, L爪, R爪]；8 维臂曲线裁前 7。"""
    root, curves_a = two_task_root
    ds = make_ds(root)
    s = ds[3]                                          # task_a/ep_001 第 3 帧
    expect_a = np.concatenate([
        curves_a["master/arm_left_position_align"][3, :7],
        curves_a["master/arm_right_position_align"][3, :7],
        curves_a["master/end_effector_left_position_align"][3],
        curves_a["master/end_effector_right_position_align"][3]])
    expect_p = np.concatenate([
        curves_a["puppet/arm_left_position_align"][3, :7],
        curves_a["puppet/arm_right_position_align"][3, :7],
        curves_a["puppet/end_effector_left_position_align"][3],
        curves_a["puppet/end_effector_right_position_align"][3]])
    np.testing.assert_allclose(s["action"].numpy(), expect_a, rtol=1e-6)
    np.testing.assert_allclose(s["proprio"].numpy(), expect_p, rtol=1e-6)
    ds.close()


def test_missing_camera_fails_loud(two_task_root):
    root, _ = two_task_root
    with pytest.raises(KeyError, match=OTHER_CAM):
        make_ds(root, camera=OTHER_CAM)


def test_color_channel_flag_respected(two_task_root):
    """channel=rgb 不反转；channel=bgr 反转——用非对称纯色钉死。"""
    root, _ = two_task_root
    ds = make_ds(root, image_size=28)                  # fixture 写 channel=rgb
    assert ds._color_channel == "rgb"
    ds.close()


def test_raw_supervision_direct():
    sup = RawDepthSupervision(d_min=10, d_max=100)
    d = np.array([[0, 10, 50], [100, 200, np.nan]], dtype=np.float32)
    out = sup(d)
    np.testing.assert_array_equal(out["depth"], d)     # 直通
    np.testing.assert_array_equal(
        out["mask"], [[0, 0, 1], [0, 0, 0]])           # 边界开区间 + NaN


def test_preprocessor_dims_and_missing_curve():
    pre = FrankaJointGripperPreprocessor()
    curves = {k: np.zeros((3, 8), np.float32) for k in ARM_KEYS}
    curves.update({k: np.zeros((3, 1), np.float32) for k in EE_KEYS})
    assert pre.dims(curves) == (16, 16)
    with pytest.raises(KeyError, match="master"):
        pre.action({"puppet/arm_left_position_align": np.zeros((3, 8))}, 0)
