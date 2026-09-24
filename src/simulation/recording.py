"""录制核心：相机随机化 + K/T 提取 + RoboMIND 子集 HDF5 落盘。

2026-09-17 sawwam 裁决（src/sawwam/guideline.md §6.3 硬前提）：
- 每集 2–3 路相机同录，各路独立随机化（pos/look_at/focal，范围见
  config/simulation.yaml 的 randomize 块），**集内 K/T 固定**；
- 精确 K/T 逐集写进 HDF5；深度语义不变（distance_to_image_plane，m→mm）。

本模块约定（sawwam dataloader 的唯一依据，禁止擅改）：
- **外参** `camera_extrinsics/<cam>` = T_base_cam（4×4，相机→基座系，
  相机轴为 ROS 约定：x 右 / y 下 / z 前）。
- **基座系**（2026-09-17 用户裁决）：两臂基座中点，旋转 = 单位阵
  （与世界系轴对齐）。运行时从 articulation root pose 实测，禁止硬编码。
- **内参** `camera_intrinsics/<cam>/matrix` = 原始像素单位 K（3×3），
  未做 letterbox 折叠——折叠进 K 是 dataloader 侧职责（guideline §2.1）。
- **夹爪**：连续闭合量（米，高=抓握，对齐 RoboMIND 语义；用户裁决：
  保连续，禁止二值化）。sim 原生指关节行程 [0,0.04] 大=张开，录制时
  线性翻转 g_out = 0.04 − g_sim；与 RoboMIND-sim 的 ~0.16 量程差由
  sawvla action.py 已裁决的归一化统计桥接，本侧不缩放。

顶层不 import isaaclab/torch（保证 pytest 无 GPU 可跑）；触及 env/传感器
的函数在函数体内延迟导入。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# PinholeCameraCfg 默认 horizontal_aperture（mm），fx=fy=W·f/HA（方形像素，
# 竖向光圈按 H/W 等比，与 IsaacLab spawn 行为一致）
HORIZONTAL_APERTURE_MM = 20.955
GRIPPER_TRAVEL_M = 0.04          # 与 mdp.DualArmAbsolutePositionActionCfg 一致
CONTROL_HZ = 15                  # 与 simulation.yaml env.decimation 对应


# --------------------------------------------------------------------------- #
# 纯数学（无 isaaclab 依赖）
# --------------------------------------------------------------------------- #

def pinhole_K(width: int, height: int, focal_mm: float) -> np.ndarray:
    """焦距 mm + 固定光圈 → 像素单位 K（3×3）。cx=W/2, cy=H/2。"""
    f = width * focal_mm / HORIZONTAL_APERTURE_MM
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def quat_wxyz_to_matrix(q) -> np.ndarray:
    """四元数 (w,x,y,z) → 旋转矩阵（3×3）。"""
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], dtype=np.float64)


def pose_to_matrix(pos, quat_wxyz) -> np.ndarray:
    """位置 + 四元数 (w,x,y,z) → 齐次变换 T（4×4， row-vec 约定 R|t）。"""
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def flip_gripper16(a16: np.ndarray) -> np.ndarray:
    """16 维向量 [L7,R7,gL,gR] 的夹爪维线性翻转：sim 大=张开 → 闭合量
    （高=抓握，RoboMIND 语义）。连续映射，不二值化（2026-09-17 用户裁决）。"""
    out = np.asarray(a16, dtype=np.float32).copy()
    out[14] = GRIPPER_TRAVEL_M - out[14]
    out[15] = GRIPPER_TRAVEL_M - out[15]
    return out


def sample_camera_poses(cam_cfgs: dict, rng: np.random.Generator
                        ) -> dict[str, dict]:
    """每相机独立采样 {pos, look_at, focal_mm}。无 randomize 块 → 原值。

    pos/look_at 各轴 ±delta 均匀采样；focal 在 [lo, hi] 均匀采样。
    """
    poses = {}
    for name, cam in cam_cfgs.items():
        pos = np.asarray(cam["pos"], dtype=np.float64)
        look = np.asarray(cam["look_at"], dtype=np.float64)
        focal = float(cam["focal_length_mm"])
        r = cam.get("randomize")
        if r:
            pos = pos + rng.uniform(-1, 1, 3) * np.asarray(
                r["pos_delta_m"], dtype=np.float64)
            look = look + rng.uniform(-1, 1, 3) * np.asarray(
                r["look_at_delta_m"], dtype=np.float64)
            lo, hi = r["focal_range_mm"]
            focal = float(rng.uniform(lo, hi))
        poses[name] = {"pos": pos, "look_at": look, "focal_mm": focal}
    return poses


# --------------------------------------------------------------------------- #
# env/传感器交互（延迟导入 isaaclab）
# --------------------------------------------------------------------------- #

def randomize_cameras(env, poses: dict[str, dict]) -> None:
    """按 sample_camera_poses 的结果设置每相机位姿 + 内参（env 0）。

    位姿走 set_world_poses_from_view（内部处理 up 轴与约定转换）；
    内参走 set_intrinsic_matrices，K 由 pinhole_K 给出，focal_length 传
    mm 数值——换算恒等（USD focal := 传入值，光圈 := f·W/fx = 20.955），
    与 spawn 配置自洽。集内只调一次（reset 后、rollout 前）。
    """
    import torch
    for name, p in poses.items():
        cam = env.unwrapped.scene.sensors[f"camera_{name}"]
        dev = env.unwrapped.device
        eyes = torch.from_numpy(np.asarray(
            [p["pos"]], dtype=np.float32)).to(dev)
        targets = torch.from_numpy(np.asarray(
            [p["look_at"]], dtype=np.float32)).to(dev)
        cam.set_world_poses_from_view(
            eyes=eyes, targets=targets,
            env_ids=torch.tensor([0], device=dev))
        w, h = cam.cfg.width, cam.cfg.height
        K = pinhole_K(w, h, p["focal_mm"])
        cam.set_intrinsic_matrices(
            matrices=torch.from_numpy(np.asarray(
                [K], dtype=np.float32)).to(dev),
            focal_length=float(p["focal_mm"]),
            env_ids=torch.tensor([0], device=dev))


def compute_base_frame(env) -> np.ndarray:
    """基座系 T_world_base（4×4）：两臂基座中点，旋转 = 单位阵。

    从 articulation root pose 实测（原始世界系，不减 env_origin——
    相机 pos_w 同为原始世界系，两侧一致）。冒烟时应打印核对 ≈(0,0.15,0.75)。
    """
    unw = env.unwrapped
    pl = unw.scene["robot_left"].data.root_link_pos_w[0].cpu().numpy()
    pr = unw.scene["robot_right"].data.root_link_pos_w[0].cpu().numpy()
    T = np.eye(4)
    T[:3, 3] = (np.asarray(pl, dtype=np.float64)
                + np.asarray(pr, dtype=np.float64)) / 2.0
    return T


def camera_extrinsics_in_base(cam, T_world_base: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray]:
    """读相机精确 (K, T_base_cam)。T_base_cam = inv(T_world_base) @
    T_world_cam（ROS 相机轴）。要求相机 cfg 开 update_latest_camera_pose，
    否则 set_world_poses 后读到的是陈旧位姿。
    """
    assert cam.cfg.update_latest_camera_pose, \
        "录制相机必须 update_latest_camera_pose=True，否则外参缓冲陈旧"
    K = cam.data.intrinsic_matrices[0].cpu().numpy().astype(np.float64)
    T_world_cam = pose_to_matrix(
        cam.data.pos_w[0].cpu().numpy(),
        cam.data.quat_w_ros[0].cpu().numpy())
    return K, np.linalg.inv(T_world_base) @ T_world_cam


def robot_base_transforms(env, T_world_base: np.ndarray
                          ) -> dict[str, np.ndarray]:
    """两臂 root 在基座系下的位姿（base_to_robot_{left,right}，对齐
    RoboMIND sim 的 base_to_robot_transformation 组）。"""
    unw = env.unwrapped
    out = {}
    for side in ("left", "right"):
        art = unw.scene[f"robot_{side}"]
        T_world_robot = pose_to_matrix(
            art.data.root_link_pos_w[0].cpu().numpy(),
            art.data.root_link_quat_w[0].cpu().numpy())
        out[side] = np.linalg.inv(T_world_base) @ T_world_robot
    return out


def absolute_joint16(env) -> np.ndarray:
    """两臂绝对关节角 → 16 维 [L7,R7,gL,gR]（puppet 实测；夹爪已翻转成
    闭合量）。与 mdp._proprio 的 default-relative 观测不同——录制要绝对值。
    """
    unw = env.unwrapped
    arm_joints = [f"panda_joint{i}" for i in range(1, 8)]
    finger = ["panda_finger_joint1", "panda_finger_joint2"]
    parts = []
    for side in ("left", "right"):
        art = unw.scene[f"robot_{side}"]
        ai, _ = art.find_joints(arm_joints, preserve_order=True)
        gi, _ = art.find_joints(finger, preserve_order=True)
        q = art.data.joint_pos[0]
        parts.append(q[ai].cpu().numpy())
        parts.append(np.array([q[gi].mean().item()], dtype=np.float64))
    a = np.concatenate(parts).astype(np.float32)   # [L7, gL, R7, gR]
    # 重排成契约顺序 [L7, R7, gL, gR]
    a = np.concatenate([a[:7], a[8:15], a[7:8], a[15:16]])
    return flip_gripper16(a)


# --------------------------------------------------------------------------- #
# RoboMIND 子集 HDF5 落盘
# --------------------------------------------------------------------------- #

def _vlen_write(f, path: str, buffers: list[bytes]) -> None:
    """vlen uint8 数据集（JPEG/PNG 字节流），与 tests/test_dataset.py 的
    参照实现同模式。"""
    import h5py
    f.require_group(path.rsplit("/", 1)[0])
    ds = f.create_dataset(path, (len(buffers),),
                          dtype=h5py.vlen_dtype(np.dtype("uint8")))
    for i, b in enumerate(buffers):
        ds[i] = np.frombuffer(b, np.uint8)


class EpisodeWriter:
    """单集缓存 + RoboMIND 子集 schema 落盘。

    schema（dataset.py 读取所依赖的键全部覆盖）：
      camera_observations/color_images/<cam>   vlen uint8 JPEG（RGB 直接编码）
      camera_observations/depth_images/<cam>   vlen uint8 PNG（uint16 mm）
      camera_observations/{timestamp,is_intervene}
      camera_color_channel/<cam>               "rgb"
      master|puppet/{arm_left,arm_right,end_effector_left,end_effector_right}
          _position_align/{data,timestamp,is_intervene}
      camera_intrinsics/<cam>/matrix           (3,3) f8，原始像素单位
      camera_extrinsics/<cam>                  (4,4) f8，T_base_cam（ROS 轴）
      camera_color_resolution|camera_depth_resolution/<cam>   (2,) = (W,H)
      base_to_robot_transformation/base_to_robot_{left,right} (4,4)
      metadata/{collection_time,collector,data_format_version,
                data_type,language_instruction,trajectory_length}
    """

    def __init__(self, camera_names: list[str]) -> None:
        self.camera_names = list(camera_names)
        self._rgb: dict[str, list[bytes]] = {c: [] for c in self.camera_names}
        self._depth: dict[str, list[bytes]] = {c: [] for c in self.camera_names}
        self._master: list[np.ndarray] = []
        self._puppet: list[np.ndarray] = []
        self._ts: list[int] = []
        self._res: dict[str, tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self._ts)

    def add_frame(self, rgb: dict[str, np.ndarray],
                  depth_mm: dict[str, np.ndarray],
                  master16: np.ndarray, puppet16: np.ndarray,
                  timestamp_us: int) -> None:
        """rgb: {cam: (H,W,3) uint8 RGB}；depth_mm: {cam: (H,W) uint16 mm}；
        master/puppet: 16 维（夹爪须已是翻转后闭合量）。"""
        import cv2
        for c in self.camera_names:
            im = np.ascontiguousarray(rgb[c][..., :3])
            assert im.dtype == np.uint8, f"{c} rgb 须 uint8，得 {im.dtype}"
            d = np.ascontiguousarray(depth_mm[c])
            assert d.dtype == np.uint16, f"{c} depth 须 uint16，得 {d.dtype}"
            self._rgb[c].append(cv2.imencode(".jpg", im)[1].tobytes())
            self._depth[c].append(cv2.imencode(".png", d)[1].tobytes())
            self._res[c] = (im.shape[1], im.shape[0])
        m = np.asarray(master16, dtype=np.float32).reshape(16)
        p = np.asarray(puppet16, dtype=np.float32).reshape(16)
        self._master.append(m)
        self._puppet.append(p)
        self._ts.append(int(timestamp_us))

    def save(self, path: Path, intrinsics: dict[str, np.ndarray],
             extrinsics: dict[str, np.ndarray],
             base_to_robot: dict[str, np.ndarray],
             language_instruction: str = "hang the cup on the holder"
             ) -> None:
        import datetime
        import h5py
        if len(self) == 0:
            raise ValueError("空 episode，拒绝落盘")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        T = len(self)
        ts = np.asarray(self._ts, dtype=np.int64)
        zeros_b = np.zeros(T, dtype=np.bool_)
        master = np.stack(self._master)     # (T,16)
        puppet = np.stack(self._puppet)

        with h5py.File(path, "w") as f:
            for c in self.camera_names:
                _vlen_write(f, f"camera_observations/color_images/{c}",
                            self._rgb[c])
                _vlen_write(f, f"camera_observations/depth_images/{c}",
                            self._depth[c])
                f.create_dataset(f"camera_color_channel/{c}",
                                 data=np.bytes_("rgb"))
                f.create_dataset(f"camera_intrinsics/{c}/matrix",
                                 data=np.asarray(intrinsics[c], np.float64))
                f.create_dataset(f"camera_extrinsics/{c}",
                                 data=np.asarray(extrinsics[c], np.float64))
                f.create_dataset(f"camera_color_resolution/{c}",
                                 data=np.asarray(self._res[c], np.int64))
                f.create_dataset(f"camera_depth_resolution/{c}",
                                 data=np.asarray(self._res[c], np.int64))
            f.create_dataset("camera_observations/timestamp", data=ts)
            f.create_dataset("camera_observations/is_intervene", data=zeros_b)

            # 16 维 [L7,R7,gL,gR] → 8 条 RoboMIND 曲线组
            for role, arr in (("master", master), ("puppet", puppet)):
                curves = {
                    "arm_left": arr[:, :7], "arm_right": arr[:, 7:14],
                    "end_effector_left": arr[:, 14:15],
                    "end_effector_right": arr[:, 15:16],
                }
                for key, v in curves.items():
                    g = f.require_group(f"{role}/{key}_position_align")
                    g.create_dataset("data", data=v.astype(np.float32))
                    g.create_dataset("timestamp", data=ts)
                    g.create_dataset("is_intervene", data=zeros_b)

            for side, M in base_to_robot.items():
                f.create_dataset(
                    f"base_to_robot_transformation/base_to_robot_{side}",
                    data=np.asarray(M, np.float64))

            meta = f.require_group("metadata")
            meta.create_dataset("collection_time", data=np.bytes_(
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
            meta.create_dataset("collector", data=np.bytes_("isaaclab_record"))
            meta.create_dataset("data_format_version", data=np.bytes_("1.0"))
            meta.create_dataset("data_type", data=np.bytes_("training"))
            meta.create_dataset("language_instruction",
                                data=np.bytes_(language_instruction))
            meta.create_dataset("trajectory_length", data=np.int64(T))
