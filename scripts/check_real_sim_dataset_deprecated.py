#!/usr/bin/env python
"""对比 real / sim 预处理后的图像与原始动作信号并可视化。

目的（2026-09-11，配合夹爪/动作空间裁决）：回答"两域动作是否本质相同
（同为关节绝对位置、同关节序、同量纲）只是表达不同（零点/符号/范围差异），
还是定义上有根本区别（如一侧是绝对位置另一侧是增量/速度）"。

产出：
1. ``<outdir>/images.png`` —— 经完整预处理管线（CenterCropPreprocessor
   方案 D + ImageNet 归一化）的 RGB 样本网格 + 原始深度/掩码；
2. ``<outdir>/actions.png`` —— 动作（master）逐关节值分布、逐步增量分布、
   master→puppet 追踪滞后与残差、样本轨迹；
3. 终端汇总表：逐关节 real/sim 范围、master-puppet 残差、滞后步数。

判读提示：
- 值分布形状相似但平移/翻转 ⇒ 表达不同（零点或符号约定），可仿射对齐；
- 增量分布一侧尖峰一侧宽散、或滞后结构一侧为 0 一侧为 1 步 ⇒ 定义可能
  不同（绝对位置 vs 增量/速度指令）；
- 滞后（puppet 落后 master 的步数）两域都稳定且小 ⇒ 两边都是"位置指令
  + 伺服追踪"语义。

用法：
    python scripts/check_real_sim_dataset.py \
        --episodes 40 --stride 5 --outdir outputs/check_dataset
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sawvla.data import RoboMindDataset  # noqa: E402
from sawvla.data.image import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

JOINTS = [f"L{i + 1}" for i in range(7)] + [f"R{i + 1}" for i in range(7)]
# 物理关节角不可能超过 2π：超出即判定为饱和/垃圾信号（2026-09-11 实测：
# real master R1 在 380/883 个 episode 整集钉死在 29–32 rad，属数据质量问题，
# 直方图与统计必须剔除，否则分布被压扁）
PHYS_RANGE = 2 * np.pi
REAL_CFG = dict(root="data/RoboMIND2.0-Tienkung", domain_id=0,
                camera="camera_top", depth_range=(1, 10000))
SIM_CFG = dict(root="data/RoboMIND2.0-Tienkung-sim", domain_id=1,
               camera="camera_head", depth_range=(1, 5000))


# --------------------------------------------------------------------------- #
def stratified_episodes(ds: RoboMindDataset, n: int, rng: np.random.Generator):
    """按任务分层抽 episode 索引（每任务至少 1 个）。"""
    by_task: dict[str, list[int]] = {}
    for i, e in enumerate(ds.episodes):
        by_task.setdefault(e["task"], []).append(i)
    per = max(1, n // len(by_task))
    picked = []
    for task in sorted(by_task):
        pool = by_task[task]
        k = min(per, len(pool))
        picked.extend(rng.choice(pool, size=k, replace=False).tolist())
    return sorted(picked)


def episode_curves(ds: RoboMindDataset, ep_idx: int):
    """读一个 episode 的 master/puppet 双臂关节曲线 → (T,14) 各一。"""
    e = ds.episodes[ep_idx]
    import h5py
    with h5py.File(e["path"], "r") as f:
        def cat(side):
            return np.concatenate(
                [f[f"{side}/arm_left_position_align/data"][:],
                 f[f"{side}/arm_right_position_align/data"][:]], axis=1)
        return cat("master").astype(np.float64), cat("puppet").astype(np.float64)


def lag_and_residual(master: np.ndarray, puppet: np.ndarray, max_lag: int = 10):
    """puppet 落后 master 的步数与归一化残差（逐关节）。

    lag>0 表示 puppet[t] ≈ master[t-lag]（指令先行、伺服追踪）。
    残差 = min_lag MSE / var(puppet)，≈0 为完美追踪。
    """
    T = master.shape[0]
    lags = np.zeros(14)
    res = np.zeros(14)
    for j in range(14):
        m, p = master[:, j], puppet[:, j]
        if np.isnan(m).any():              # 饱和关节：滞后/残差无意义
            lags[j], res[j] = np.nan, np.nan
            continue
        var = p.var() + 1e-12
        best = (np.inf, 0)
        for lag in range(-2, max_lag + 1):
            if lag >= 0:
                a, b = m[:T - lag or None], p[lag:]
            else:
                a, b = m[-lag:], p[:T + lag]
            if len(a) < 10:
                continue
            err = np.mean((a - b) ** 2) / var
            if err < best[0]:
                best = (err, lag)
        res[j], lags[j] = best
    return lags, res


def collect(domain_ds: RoboMindDataset, ep_indices, stride: int):
    masters, puppets, lags_l, res_l, trajs = [], [], [], [], []
    n_sat_ep = np.zeros(14, dtype=int)     # 逐关节：含饱和帧的 episode 数
    for ep in ep_indices:
        m, p = episode_curves(domain_ds, ep)
        sat = np.abs(m) > PHYS_RANGE       # (T, 14) 饱和帧掩码
        n_sat_ep += sat.any(axis=0)
        m_clean = np.where(sat, np.nan, m)
        masters.append(m_clean[::stride])
        puppets.append(p[::stride])
        l, r = lag_and_residual(m_clean, p)
        lags_l.append(l)
        res_l.append(r)
        trajs.append((m, p))
    return {
        "master": np.concatenate(masters),
        "puppet": np.concatenate(puppets),
        "lags": np.stack(lags_l),
        "res": np.stack(res_l),
        "trajs": trajs,
        "sat_episodes": n_sat_ep,
        "n_episodes": len(ep_indices),
    }


# --------------------------------------------------------------------------- #
def plot_images(real_ds, sim_ds, real_eps, sim_eps, out: Path, k: int = 4):
    fig, axes = plt.subplots(4, k, figsize=(4.2 * k, 15))
    for row, (ds, eps, tag) in enumerate(
            [(real_ds, real_eps, "real"), (sim_ds, sim_eps, "sim")]):
        for col in range(k):
            ep = eps[col % len(eps)]
            gidx = int(ds._cum[ep]) + ds._frames_per_ep[ep] // 2
            s = ds[gidx]
            rgb = (s["rgb"].numpy().transpose(1, 2, 0) * IMAGENET_STD
                   + IMAGENET_MEAN).clip(0, 1)
            depth = s["depth"].numpy()
            mask = s["mask"].numpy()
            d_show = np.where(mask > 0, depth, np.nan)

            ax = axes[2 * row, col]
            ax.imshow(rgb)
            ax.set_title(f"{tag} RGB (preprocessed)\n{s['episode_name']}",
                         fontsize=8)
            ax.axis("off")

            axd = axes[2 * row + 1, col]
            im = axd.imshow(d_show, cmap="turbo")
            axd.set_title(f"{tag} depth raw (mask valid "
                          f"{mask.mean():.0%})", fontsize=8)
            axd.axis("off")
            fig.colorbar(im, ax=axd, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")


def plot_actions(real, sim, out: Path):
    fig = plt.figure(figsize=(26, 22))
    gs = fig.add_gridspec(6, 7, hspace=0.55, wspace=0.3)

    def hist_row(row0, getter, title, bins=80):
        for j in range(14):
            ax = fig.add_subplot(gs[row0 + j // 7, j % 7])
            r, s = getter(real)[..., j], getter(sim)[..., j]
            r, s = r[~np.isnan(r)], s[~np.isnan(s)]
            lo = min(np.percentile(r, 0.5), np.percentile(s, 0.5))
            hi = max(np.percentile(r, 99.5), np.percentile(s, 99.5))
            ax.hist(r, bins=bins, range=(lo, hi), density=True, alpha=0.55,
                    color="tab:blue", label="real")
            ax.hist(s, bins=bins, range=(lo, hi), density=True, alpha=0.55,
                    color="tab:orange", label="sim")
            ax.set_title(f"{title} {JOINTS[j]}", fontsize=9)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.legend(fontsize=8)

    # 行 1-2：master 值分布；行 3-4：master 逐步增量分布
    hist_row(0, lambda d: d["master"], "value")
    hist_row(2, lambda d: np.diff(d["master"], axis=0), "delta")

    # 行 5：滞后与残差（跨 episode 中位数）
    ax = fig.add_subplot(gs[4, :4])
    x = np.arange(14)
    ax.bar(x - 0.2, np.nanmedian(real["lags"], 0), 0.4, label="real")
    ax.bar(x + 0.2, np.nanmedian(sim["lags"], 0), 0.4, label="sim")
    ax.set_xticks(x, JOINTS)
    ax.set_title("puppet lag behind master (steps, median over episodes)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[4, 4:])
    ax.bar(x - 0.2, np.nanmedian(real["res"], 0), 0.4, label="real")
    ax.bar(x + 0.2, np.nanmedian(sim["res"], 0), 0.4, label="sim")
    ax.set_xticks(x, JOINTS)
    ax.set_yscale("log")
    ax.set_title("master→puppet tracking residual / var (log)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 行 6：样本轨迹（各取第一个 episode 的左右臂 j0）
    for c, (d, tag) in enumerate([(real, "real"), (sim, "sim")]):
        ax = fig.add_subplot(gs[5, c * 3:(c + 1) * 3])
        m, p = d["trajs"][0]
        for j, cj in [(0, "tab:blue"), (7, "tab:red")]:
            ax.plot(m[:, j], color=cj, ls="-", lw=1,
                    label=f"master {JOINTS[j]}")
            ax.plot(p[:, j], color=cj, ls="--", lw=1, alpha=0.7,
                    label=f"puppet {JOINTS[j]}")
        ax.set_title(f"{tag} sample trajectory (ep 0)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[5, 6])
    ax.axis("off")
    ax.text(0, 0.9, "How to read", fontsize=10, weight="bold")
    ax.text(0, 0.1, "value hists shifted/flipped\n=> convention difference\n"
                    "delta shape differs\n=> definition suspect\n"
                    "small stable lag\n=> position-cmd semantics",
            fontsize=8, va="bottom")

    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")


def print_summary(real, sim):
    for tag, d in [("real", real), ("sim", sim)]:
        sat = {JOINTS[j]: int(d["sat_episodes"][j])
               for j in range(14) if d["sat_episodes"][j] > 0}
        print(f"[{tag}] episodes with saturated joints (|master|>2pi, "
              f"excluded from hists): {sat or 'none'} "
              f"/ {d['n_episodes']} episodes")
    hdr = (f"{'joint':>5} | {'real range':^22} | {'sim range':^22} | "
           f"{'resid r/s':^14} | {'lag r/s':^8}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for j in range(14):
        rm, rp = real["master"][:, j], sim["master"][:, j]
        print(f"{JOINTS[j]:>5} | "
              f"[{np.nanpercentile(rm, 1):7.3f},{np.nanpercentile(rm, 99):7.3f}]   | "
              f"[{np.nanpercentile(rp, 1):7.3f},{np.nanpercentile(rp, 99):7.3f}]   | "
              f"{np.nanmedian(real['res'][:, j]):7.4f}/{np.nanmedian(sim['res'][:, j]):7.4f} | "
              f"{np.nanmedian(real['lags'][:, j]):4.0f}/{np.nanmedian(sim['lags'][:, j]):4.0f}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=40,
                    help="每域分层抽样 episode 数（默认 40）")
    ap.add_argument("--stride", type=int, default=5,
                    help="曲线抽帧步长（默认 5）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/check_dataset"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    real_ds = RoboMindDataset(**REAL_CFG)
    sim_ds = RoboMindDataset(**SIM_CFG)
    real_eps = stratified_episodes(real_ds, args.episodes, rng)
    sim_eps = stratified_episodes(sim_ds, args.episodes, rng)
    print(f"real: {len(real_eps)} episodes over "
          f"{len({real_ds.episodes[i]['task'] for i in real_eps})} tasks; "
          f"sim: {len(sim_eps)} episodes over "
          f"{len({sim_ds.episodes[i]['task'] for i in sim_eps})} tasks")

    plot_images(real_ds, sim_ds, real_eps, sim_eps, args.outdir / "images.png")

    real = collect(real_ds, real_eps, args.stride)
    sim = collect(sim_ds, sim_eps, args.stride)
    print_summary(real, sim)
    plot_actions(real, sim, args.outdir / "actions.png")

    real_ds.close()
    sim_ds.close()


if __name__ == "__main__":
    main()
