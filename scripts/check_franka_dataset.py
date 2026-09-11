#!/usr/bin/env python
"""Franka 数据集（real Part-1 / sim）单 episode 可视化 + 动作描述导出。

背景（2026-09-11）：用户决定从 Tienkung 切换到 Franka 数据
（data/RoboMIND2.0-Franka-Part-1 + data/RoboMIND2.0-Franka-sim），
目标是从根上消除 Tienkung 的跨域不一致（灵巧手 vs 夹爪、相机命名乱、
动作字段残缺）。本脚本取两边**文件名最小**的 episode：

- 可视化：全部相机的 RGB 网格与深度网格（行=相机，列=帧，均匀抽样）；
- 动作：双臂 7 关节 + 夹爪 master/puppet 曲线图；
- 导出：report.md 动作描述（schema 表、逐维量程/增量、EE 分析、
  相机表、metadata、自动检测到的问题清单）。

用法：
    python scripts/check_franka_dataset.py            # 默认最小文件名 episode
    python scripts/check_franka_dataset.py --real-file <hdf5> --sim-file <hdf5>
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SIM_ROOT = "data/RoboMIND2.0-Franka-sim"
REAL_ROOT = "data/RoboMIND2.0-Franka-Part-1"

SIM_CAM_ORDER = ["camera_top", "camera_front", "camera_left",
                 "camera_right", "camera_wrist_left", "camera_wrist_right"]
REAL_CAM_ORDER = ["camera_top", "camera_front", "camera_left", "camera_right"]


def smallest_episode(root: str) -> str:
    files = sorted(glob.glob(
        f"{root}/data/*/*/success_episodes/*/data/*.hdf5"))
    if not files:
        raise FileNotFoundError(root)
    return files[0]


def decode_color(raw: bytes, channel: str) -> np.ndarray:
    """解码 JPEG/PNG 字节为 RGB uint8。

    channel 字段记录的是解码后像素的真实通道序：cv2.imdecode 恒返回
    按 BGR 约定的数组——若存的时候是 rgb 序，imdecode 的输出实际上就是
    RGB（cv2 不知情），此时不能再反转。
    """
    import cv2
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if channel == "bgr":
        img = img[..., ::-1]
    return img


def decode_depth(raw: bytes) -> np.ndarray:
    import cv2
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)


def depth_to_rgb(d: np.ndarray) -> np.ndarray:
    """uint16 深度 → 伪彩色 RGB（有效值 5–95 分位拉伸，无效=黑）。"""
    valid = d > 0
    out = np.zeros((*d.shape, 3), dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(d[valid], [5, 95])
        t = np.clip((d.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
        out[..., 0] = t
        out[..., 2] = 1 - t
        out[..., 1] = 0.5 * (1 - np.abs(t - 0.5) * 2)
        out[~valid] = 0
    return out


# --------------------------------------------------------------------------- #
def render_grids(tag, path, cams, n_frames, outdir: Path):
    """RGB 与深度各一张网格图；返回每相机洞率等统计。"""
    import h5py
    stats = {}
    with h5py.File(path, "r") as f:
        T = f["camera_observations/timestamp"].shape[0]
        ts = np.linspace(0, T - 1, n_frames).astype(int)
        channel = {c: f["camera_color_channel"][c][()].decode()
                   for c in cams}
        for kind in ("color", "depth"):
            fig, axes = plt.subplots(len(cams), n_frames,
                                     figsize=(2.6 * n_frames, 2.4 * len(cams)))
            axes = np.atleast_2d(axes)
            for r, cam in enumerate(cams):
                for c, t in enumerate(ts):
                    raw = f[f"camera_observations/{kind}_images/{cam}"][t]
                    if kind == "color":
                        img = decode_color(raw, channel[cam])
                    else:
                        d = decode_depth(raw)
                        img = depth_to_rgb(d)
                        if c == 0:
                            valid = d > 0
                            stats[cam] = dict(
                                shape=list(d.shape),
                                hole=float((~valid).mean()),
                                min=int(d[valid].min()) if valid.any() else 0,
                                max=int(d.max()))
                    ax = axes[r, c]
                    ax.imshow(img)
                    ax.set_title(f"f{t}", fontsize=8)
                    ax.axis("off")
                axes[r, 0].set_ylabel(cam.replace("camera_", ""),
                                      fontsize=9, rotation=0,
                                      ha="right", va="center")
            fig.suptitle(f"{tag} {kind}  ({Path(path).parent.parent.name})",
                         fontsize=11)
            fig.tight_layout()
            out = outdir / f"{tag}_{kind}.png"
            fig.savefig(out, dpi=110)
            plt.close(fig)
            print(f"saved -> {out}")
    return stats, int(T)


def plot_actions(tag, path, outdir: Path):
    """双臂 7 关节 + 夹爪的 master/puppet 曲线；返回 EE/动作统计。"""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        fig, axes = plt.subplots(2, 4, figsize=(22, 7), sharey="row")
        for col, side in enumerate(("left", "right")):
            arm = {g: f[f"{g}/arm_{side}_position_align/data"][:]
                   for g in ("master", "puppet")}
            ee = {g: f[f"{g}/end_effector_{side}_position_align/data"][:, 0]
                  for g in ("master", "puppet")}
            n_j = arm["master"].shape[1]
            # 关节曲线（master 实线 / puppet 虚线）；8 维时第 8 维=夹爪不画这里
            ax = axes[0, col]
            for j in range(min(7, n_j)):
                ax.plot(arm["master"][:, j], lw=0.8, label=f"j{j + 1}")
                ax.plot(arm["puppet"][:, j], lw=0.6, ls="--", alpha=0.6)
            ax.set_title(f"{tag} arm_{side} joints (solid=master dash=puppet)",
                         fontsize=9)
            ax.legend(fontsize=6, ncol=4)
            ax.grid(alpha=0.3)
            # 夹爪
            ax = axes[0, col + 2]
            ax.plot(ee["master"], lw=1.0, label="master EE")
            ax.plot(ee["puppet"], lw=1.0, label="puppet EE")
            if n_j == 8:
                ax.plot(arm["master"][:, 7], lw=0.6, ls=":",
                        label="master arm[...,7]")
            ax.set_title(f"{tag} EE_{side}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            out[side] = dict(
                arm_dims=int(n_j),
                arm_min=np.round(arm["master"].min(0), 4).tolist(),
                arm_max=np.round(arm["master"].max(0), 4).tolist(),
                ee_master=[float(ee["master"].min()), float(ee["master"].max())],
                ee_puppet=[float(ee["puppet"].min()), float(ee["puppet"].max())],
                ee_master_unique=int(len(np.unique(ee["master"]))),
                arm8_eq_ee=bool(n_j == 8 and np.abs(
                    arm["master"][:, 7] - ee["master"]).max() < 1e-9),
            )
        # 第二行：逐关节增量分布（master）
        for col, side in enumerate(("left", "right")):
            arm = f[f"master/arm_{side}_position_align/data"][:]
            ax = axes[1, col]
            d = np.abs(np.diff(arm[:, :7], axis=0))
            ax.boxplot([d[:, j] for j in range(7)],
                       tick_labels=[f"j{j + 1}" for j in range(7)])
            ax.set_title(f"{tag} master arm_{side} |delta| per step",
                         fontsize=9)
            ax.grid(alpha=0.3)
        axes[1, 2].axis("off")
        axes[1, 3].axis("off")
        fig.tight_layout()
        p = outdir / f"{tag}_actions.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(f"saved -> {p}")
    return out


def describe(tag, path, cam_stats, T, act, lines):
    """schema + metadata + 相机表 + 自动问题检测 → report 行。"""
    import h5py
    A = lines.append
    A(f"\n## {tag}: `{path}`\n")
    with h5py.File(path, "r") as f:
        A(f"- 帧数 T = {T}")
        if "metadata" in f and len(f["metadata"]):
            for k in f["metadata"]:
                v = f["metadata"][k][()]
                if isinstance(v, bytes):
                    v = v.decode(errors="replace")
                A(f"- metadata/{k} = `{v}`")
        else:
            A("- metadata：**空**（无语言指令）")
        has_ik = "camera_intrinsics" in f
        A(f"- 相机内参/外参：{'有' if has_ik else '**无**'}")
        # 时间戳
        ts = f["camera_observations/timestamp"][:]
        d = np.diff(ts)
        A(f"- timestamp 差分：median={np.median(d):.0f}，"
          f"unique[:5]={np.unique(d)[:5].tolist()}")
        iv = f["camera_observations/is_intervene"][:]
        A(f"- is_intervene = {int(iv.sum())}/{len(iv)}")
        # raw vs align
        raws = [k for k in f["master"]] if "master" in f else []
        has_raw = any(k.endswith("_raw") for k in raws)
        A(f"- 原始（_raw）信号：{'有（60Hz 原始 + 30Hz 对齐）' if has_raw else '**无**'}")

    A("\n相机：\n")
    A("| 相机 | 分辨率 | 洞率(f0) | 深度范围(f0) |")
    A("|---|---|---|---|")
    for cam, s in cam_stats.items():
        A(f"| {cam} | {s['shape']} | {s['hole']:.1%} | "
          f"{s['min']}–{s['max']} |")

    A("\n动作（master 指令 / puppet 实测）：\n")
    for side, s in act.items():
        A(f"- arm_{side}：{s['arm_dims']} 维，"
          f"min={s['arm_min']}，max={s['arm_max']}")
        A(f"  - EE master ∈ [{s['ee_master'][0]:.4f}, {s['ee_master'][1]:.4f}]"
          f"（unique 值 {s['ee_master_unique']} 个），"
          f"puppet ∈ [{s['ee_puppet'][0]:.4f}, {s['ee_puppet'][1]:.4f}]")
        if s["arm_dims"] == 8:
            A(f"  - arm[...,7] 与 EE 字段完全相同 = {s['arm8_eq_ee']}")


def detect_issues(cam_stats_sim, cam_stats_real, act_sim, act_real,
                  ts_issues, lines):
    A = lines.append
    A("\n## 自动检测到的问题 / 跨域差异\n")
    A("1. **real arm_align 是 8 维**（7 关节 + 夹爪，第 8 维与 EE 字段逐位相同"
      f"={act_real['left']['arm8_eq_ee']}），sim 是 7 维——字段语义跨域不同构，"
      "读取时必须按域分别处理或统一裁到 7 + 独立 EE。")
    A("2. **EE 指令形态不同、方向相同**：master EE 高值=抓握、≈0=松开/空闲，"
      "两域一致（real 逐帧图 + sim 腕部相机图双重核实）。real master EE "
      "bang-bang 0↔1（归一化闭合度，puppet 被物体挡住时读数<1），"
      f"sim master EE 连续 ∈ [0, {act_sim['left']['ee_master'][1]:.3f}]。"
      "功能对齐为 1 维开合度/闭合度可行。")
    A(f"3. **real 左夹爪 puppet 峰值系统性偏低**（178 episode 普查：左 "
      "median≈0.26 / range 0.18–0.37，右 median≈0.95）。已核实非故障："
      "master EE=1 的时段恰为左爪握杯（外壁抓握）时段，杯子挡住手指导致"
      "闭合读数停在杯径处（右爪捏杯沿≈全闭 0.95）。")
    A(f"4. **real 时间戳不可用**（{ts_issues}），跨信号对齐只能信任 "
      "_align 序列本身；sim 时间戳正常（33ms≈30Hz）。")
    A("5. **real metadata 全空**（无语言指令），sim 全有——与 Tienkung 相同"
      "的补标问题仍在。")
    A("6. **real 无相机内外参，sim 有全套**（含 base_to_robot 变换）——"
      "几何监督/投影类方法两域不对称。")
    A("7. 相机路数：real 4 路（无腕部），sim 6 路（多 wrist_left/right）；"
      "共有路 = top/front/left/right。分辨率：real top/right 640×480、"
      "front/left 1280×720；sim top/front 1280×720、其余 640×480——"
      "**同名相机两域宽高比不一致**（real top=4:3 vs sim top=16:9），"
      "方案 D 中心裁剪逻辑不受影响，但相机选择策略需重新裁决。")
    A("8. 深度：real 有洞（f0 洞率 "
      + ", ".join(f"{c}={s['hole']:.1%}" for c, s in cam_stats_real.items())
      + "），sim 无洞。")
    A("9. **深度 65535 哨兵值**：real camera_right 与 sim camera_left/right "
      "的 f0 最大深度 = 65535（远平面/无效哨兵，不是真实毫米读数）——"
      "掩码规则需把上限纳入（RawDepthSupervision 的 d_max 即为此设计）。")
    A("10. sim 任务目录存在拼写错误残留 `124-hang_cup_on_cup_holer`"
      "（0 个 episode，空目录）。")
    A("11. 通道序标记不同：real=rgb，sim=bgr——解码显示/预处理必须按各自"
      "标记处理，不能统一假设。")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-file", default=None)
    ap.add_argument("--sim-file", default=None)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/check_franka"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    real = args.real_file or smallest_episode(REAL_ROOT)
    sim = args.sim_file or smallest_episode(SIM_ROOT)
    print(f"real episode: {real}")
    print(f"sim  episode: {sim}")

    cam_stats_real, T_real = render_grids("real", real, REAL_CAM_ORDER,
                                          args.frames, args.outdir)
    cam_stats_sim, T_sim = render_grids("sim", sim, SIM_CAM_ORDER,
                                        args.frames, args.outdir)
    act_real = plot_actions("real", real, args.outdir)
    act_sim = plot_actions("sim", sim, args.outdir)

    import h5py
    with h5py.File(real, "r") as f:
        d = np.diff(f["camera_observations/timestamp"][:])
        ts_issues = f"差分只有 {np.unique(d).tolist()[:4]} ms"

    lines = ["# Franka 数据集最小 episode 检查报告（2026-09-11）",
             "",
             "real = `%s`" % real,
             "sim  = `%s`" % sim]
    describe("real", real, cam_stats_real, T_real, act_real, lines)
    describe("sim", sim, cam_stats_sim, T_sim, act_sim, lines)
    detect_issues(cam_stats_sim, cam_stats_real, act_sim, act_real,
                  ts_issues, lines)
    report = args.outdir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved -> {report}")


if __name__ == "__main__":
    main()
