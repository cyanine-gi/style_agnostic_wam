#!/usr/bin/env python
"""sim 灵巧手 / real 夹爪 EE 信号降维分析（PCA + 相关性 + 轨迹）。

背景（2026-09-11）：sim 端执行器是灵巧手（puppet EE position 12 维/手，
master EE 6 维/手——疑似 6 电机欠驱动、12 关节测量），real 端是夹爪
（EE position 标量/手）。要回答的问题：**sim 手指运动的有效维数是多少**——
若 ≈1（或低维且主成分语义是"开合"），则两域 EE 动作可功能对齐到
每手 1 维开合度，action 从 14 升到 16，跨本体裂口闭合；
若是高维灵巧运动，则需另议（见项目记忆 / 讨论记录）。

产出：
1. ``<outdir>/ee_pca.png`` —— PCA 谱（方差占比 + 累积）、12×12 关节相关性
   热图、PC1/PC2 散点与样本轨迹；
2. 终端汇总：各信号前若干主成分方差占比、累积 90/99% 所需维数、
   参与率（participation ratio，有效维数估计）。

用法：
    python scripts/check_sim_ee_pca.py --episodes-per-task 3 \
        --outdir outputs/check_ee_pca
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SIM_ROOT = "data/RoboMIND2.0-Tienkung-sim"
REAL_ROOT = "data/RoboMIND2.0-Tienkung"

SIGNALS = {                          # 键 → (hdf5 路径, 标注)
    "puppet_L": ("puppet/end_effector_left_position_align/data", "puppet L (12d)"),
    "puppet_R": ("puppet/end_effector_right_position_align/data", "puppet R (12d)"),
    "master_L": ("master/end_effector_left_position_align/data", "master L (6d)"),
    "master_R": ("master/end_effector_right_position_align/data", "master R (6d)"),
}


# --------------------------------------------------------------------------- #
def collect_sim(episodes_per_task: int, stride: int, seed: int):
    """分层抽样 episode，读 4 路 EE 曲线。"""
    import h5py
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(
        f"{SIM_ROOT}/data/*/*/success_episodes/*/data/*.hdf5"))
    by_task: dict[str, list[str]] = {}
    for p in files:
        by_task.setdefault(p.split("/success_episodes/")[0].split("/")[-1],
                           []).append(p)
    data = {k: [] for k in SIGNALS}
    trajs = []
    n_eps = 0
    for task in sorted(by_task):
        picked = rng.choice(by_task[task],
                            size=min(episodes_per_task, len(by_task[task])),
                            replace=False)
        for p in picked:
            with h5py.File(p, "r") as f:
                arrs = {}
                for k, (path, _) in SIGNALS.items():
                    if path not in f:
                        raise KeyError(f"{p} 缺少 {path}")
                    arrs[k] = f[path][:].astype(np.float64)
                n = min(a.shape[0] for a in arrs.values())
                for k in SIGNALS:
                    data[k].append(arrs[k][:n:stride])
                trajs.append((task, {k: arrs[k][:n] for k in SIGNALS}))
                n_eps += 1
    return {k: np.concatenate(v) for k, v in data.items()}, trajs, n_eps


def collect_real(stride: int, seed: int, n_files: int = 200):
    """real 夹爪标量（对照）。

    2026-09-11 实测 real EE 覆盖不均：tidy_desktop 为 (T,) 标量、
    open_pot_lid 为 (T,1)、insert_pens / put_egg_into_box 为 (0,) 空——
    空字段跳过并在返回中报告缺失 episode 数。
    """
    import h5py
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(
        f"{REAL_ROOT}/data/*/*/success_episodes/*/data/*.hdf5"))
    picked = rng.choice(files, size=min(n_files, len(files)), replace=False)
    vals, missing = [], 0
    for side in ("left", "right"):
        per = []
        for p in picked:
            with h5py.File(p, "r") as f:
                d = f[f"puppet/end_effector_{side}_position_align/data"]
                a = d[:]
                if a.size == 0:
                    if side == "left":
                        missing += 1
                    continue
                per.append(a.reshape(-1).astype(np.float64)[::stride])
        vals.append(np.concatenate(per) if per else np.array([]))
    return vals, missing, len(picked)


def pca(x: np.ndarray):
    """标准化 PCA，返回 (方差占比, 累积, 主成分方向, 投影)。"""
    x = x[np.isfinite(x).all(axis=1)]
    mu, sd = x.mean(0), x.std(0) + 1e-12
    z = (x - mu) / sd
    cov = np.cov(z.T)
    w, v = np.linalg.eigh(cov)
    w, v = w[::-1], v[:, ::-1]
    ratio = w / w.sum()
    return ratio, np.cumsum(ratio), v, z @ v


def participation_ratio(ratio: np.ndarray) -> float:
    return float(1.0 / np.sum(ratio ** 2))


def dims_for(cum: np.ndarray, th: float) -> int:
    return int(np.searchsorted(cum, th) + 1)


# --------------------------------------------------------------------------- #
def plot(data, trajs, out: Path):
    fig = plt.figure(figsize=(26, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    pcas = {}
    # 行 1：PCA 谱（puppet / master 各一图，L/R 同图对比）
    for col, keys in enumerate([("puppet_L", "puppet_R"),
                                ("master_L", "master_R")]):
        ax = fig.add_subplot(gs[0, col])
        for k in keys:
            ratio, cum, v, _ = pca(data[k])
            pcas[k] = (ratio, cum, v)
            x = np.arange(1, len(ratio) + 1)
            ax.bar(x, ratio, alpha=0.45, label=f"{SIGNALS[k][1]} ratio")
            ax.plot(x, cum, marker="o", ms=3,
                    label=f"{SIGNALS[k][1]} cum (PR={participation_ratio(ratio):.1f})")
        ax.axhline(0.9, color="gray", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_title("PCA spectrum (standardized)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # 行 1 右两格：puppet 12×12 相关性热图
    for col, k in zip((2, 3), ("puppet_L", "puppet_R")):
        ax = fig.add_subplot(gs[0, col])
        x = data[k]
        x = x[np.isfinite(x).all(axis=1)]
        corr = np.corrcoef((x - x.mean(0)) / (x.std(0) + 1e-12), rowvar=False)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"{SIGNALS[k][1]} joint correlation")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # 行 2：PC1/PC2 散点（前 3 个 episode 投影，着色=帧序）
    for col, k in enumerate(("puppet_L", "puppet_R", "master_L", "master_R")):
        ax = fig.add_subplot(gs[1, col])
        ratio, cum, v = pcas.get(k, pca(data[k]))
        pcas.setdefault(k, (ratio, cum, v))
        for task, tr in trajs[:3]:
            z = (tr[k] - data[k].mean(0)) / (data[k].std(0) + 1e-12)
            proj = z @ v[:, :2]
            ax.scatter(proj[:, 0], proj[:, 1], s=2,
                       c=np.arange(len(proj)), cmap="viridis")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{SIGNALS[k][1]} projection (3 eps, color=time)")
        ax.grid(alpha=0.3)

    # 行 3：样本 episode 的 PC1 轨迹（puppet L/R 各一）+ real 夹爪轨迹占位
    for col, k in zip((0, 1), ("puppet_L", "puppet_R")):
        ax = fig.add_subplot(gs[2, col])
        ratio, cum, v = pcas[k]
        for task, tr in trajs[:5]:
            z = (tr[k] - data[k].mean(0)) / (data[k].std(0) + 1e-12)
            ax.plot(z @ v[:, 0], lw=0.8, label=task.split("-")[0])
        ax.set_title(f"{SIGNALS[k][1]} PC1 trajectories "
                     f"({ratio[0]:.0%} var)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[2, 2])
    for task, tr in trajs[:5]:
        d = tr["puppet_L"]
        ax.plot(d.std(axis=1), lw=0.8)     # 12 维瞬时离散度参考线
    ax.set_title("puppet L per-frame joint spread (ref)")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    ax.text(0, 0.9, "How to read", fontsize=10, weight="bold")
    ax.text(0, 0.05,
            "cum hits 90% at 1-2 dims\n"
            "  => functionally a gripper: alignable to real\n"
            "cum needs 4+ dims\n"
            "  => true dexterous motion: separate discussion\n"
            "corr heatmap all-red\n"
            "  => fingers move as one (underactuated)",
            fontsize=9, va="bottom")

    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")


def print_summary(data, real_vals, real_missing, real_total):
    print("\n=== sim EE PCA (standardized) ===")
    for k, (_, label) in SIGNALS.items():
        ratio, cum, _, _ = pca(data[k])
        top = " ".join(f"{r:.1%}" for r in ratio[:6])
        print(f"{label:>16}: top6 [{top}]  "
              f"dims@90%={dims_for(cum, 0.9)} dims@99%={dims_for(cum, 0.99)} "
              f"PR={participation_ratio(ratio):.2f}")
    print("\n=== real gripper scalar (reference) ===")
    print(f"(EE field empty in {real_missing}/{real_total} sampled episodes)")
    for side, v in zip(("left", "right"), real_vals):
        if len(v):
            print(f"{side:>16}: n={len(v)} range=[{v.min():.3f},{v.max():.3f}] "
                  f"std={v.std():.4f}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes-per-task", type=int, default=3)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/check_ee_pca"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    data, trajs, n_eps = collect_sim(args.episodes_per_task, args.stride,
                                     args.seed)
    print(f"sim: {n_eps} episodes, " +
          ", ".join(f"{k}={v.shape}" for k, v in data.items()))
    real_vals, real_missing, real_total = collect_real(args.stride, args.seed)
    print_summary(data, real_vals, real_missing, real_total)
    plot(data, trajs, args.outdir / "ee_pca.png")


if __name__ == "__main__":
    main()
