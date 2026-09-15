"""Stage 1 单测：register 递推 T、null 本体感、抽帧、latent 缓存与 clip、
选项 A 梯度口径。

不依赖真实数据与真实权重（tiny 合成张量 + 合成 HDF5 episode）。
"""

import numpy as np
import pytest
import torch

from sawvla.data import (LatentClipDataset, Stage0MotionThinnedDataset,
                         build_cache, write_index)
from sawvla.models import ActionAdapter, TransitionModel
from sawvla.signals import SignalContext, build_signals
from test_dataset import make_episode

D, G, R = 32, 4, 4          # tiny：d=32, grid=4 (16 patch), n_reg=4
ROLES = {"domain": [0, 1], "agnostic": [2, 3]}


def make_t(depth=2):
    torch.manual_seed(0)
    adapter = ActionAdapter(action_dim=16, proprio_dim=16, d=D)
    T = TransitionModel(d=D, depth=depth, n_heads=4, grid_size=G,
                        max_frames=16, n_reg=R)
    return adapter, T


def inputs(B=2, k=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, k, G * G, D, generator=g),
            torch.randn(B, k, R, D, generator=g),
            torch.randn(B, k, 16, generator=g),
            torch.randn(B, k, 16, generator=g))


# --------------------------------------------------------------------------- #
# register 递推 T
# --------------------------------------------------------------------------- #

def test_registers_forward_shape():
    adapter, T = make_t()
    z, r, a, p = inputs(k=3)
    out = T.forward_with_registers(z, r, *adapter(a, p))
    assert out["patch"].shape == z.shape
    assert out["registers"].shape == r.shape


def test_registers_identity_at_start():
    """双头零初始化 ⇒ 初始 ẑ≡z、r̂≡r（残差恒等起步）。"""
    adapter, T = make_t()
    z, r, a, p = inputs()
    out = T.forward_with_registers(z, r, *adapter(a, p))
    assert torch.allclose(out["patch"], z, atol=1e-5)
    assert torch.allclose(out["registers"], r, atol=1e-5)


def test_registers_causal_leakage():
    """因果泄漏钉死（register 模式）：改未来动作，过去预测逐位不变。"""
    adapter, T = make_t(depth=3)
    with torch.no_grad():      # 打破恒等起步，否则输出恒等于输入无法检验
        T.head.weight.normal_(0, 0.02)
        T.head_reg.weight.normal_(0, 0.02)
    z, r, a, p = inputs(B=1, k=4)
    a2 = a.clone()
    a2[0, 3] += 5.0
    out1 = T.forward_with_registers(z, r, *adapter(a, p))
    out2 = T.forward_with_registers(z, r, *adapter(a2, p))
    for key in ("patch", "registers"):
        assert torch.equal(out1[key][:, :3], out2[key][:, :3])
        assert not torch.allclose(out1[key][:, 3], out2[key][:, 3])


def test_null_proprio_ignores_future_proprio():
    """proprio_valid=False 的步：本体感槽位填 null，改其 proprio 值无影响。"""
    adapter, T = make_t(depth=2)
    with torch.no_grad():
        T.head.weight.normal_(0, 0.02)
    z, r, a, p = inputs(B=1, k=3)
    pv = torch.tensor([[True, True, False]])
    p2 = p.clone()
    p2[0, 2] += 100.0                        # 第 2 步本体感剧变
    out1 = T.forward_with_registers(z, r, *adapter(a, p), proprio_valid=pv)
    out2 = T.forward_with_registers(z, r, *adapter(a, p2), proprio_valid=pv)
    assert torch.equal(out1["patch"], out2["patch"])
    assert torch.equal(out1["registers"], out2["registers"])
    # 但 valid=True 时 proprio 必须影响输出（sanity）
    out3 = T.forward_with_registers(z, r, *adapter(a, p2))
    assert not torch.allclose(out1["patch"], out3["patch"])


def test_unroll_with_registers_matches_forward():
    adapter, T = make_t()
    z, r, a, p = inputs(B=2, k=4)
    act_tok, _ = adapter(a, p)
    out = T.unroll_with_registers(z[:, 0], r[:, 0], act_tok)
    assert out["patch"].shape == z.shape
    assert out["registers"].shape == r.shape
    # rollout 第 1 步与单步前向一致（proprio 全 null 口径对齐）
    one = T.forward_with_registers(
        z[:, :1], r[:, :1], *adapter(a[:, :1], p[:, :1]),
        proprio_valid=torch.zeros(2, 1, dtype=torch.bool))
    assert torch.allclose(out["patch"][:, 0], one["patch"][:, 0], atol=1e-5)


def test_patch_only_mode_unchanged_and_guarded():
    """n_reg=0 保持旧行为；register 模式下旧 forward 明确报错。"""
    adapter, T = make_t()
    z, r, a, p = inputs(k=2)
    with pytest.raises(RuntimeError, match="n_reg>0"):
        T(z, *adapter(a, p))
    T0 = TransitionModel(d=D, depth=1, n_heads=4, grid_size=G, n_reg=0)
    with pytest.raises(RuntimeError, match="n_reg=0"):
        T0.forward_with_registers(z, r, *adapter(a, p))


# --------------------------------------------------------------------------- #
# 抽帧（sim 30fps → 15fps）
# --------------------------------------------------------------------------- #

def test_subsample_without_thinning(tmp_path):
    """tau=0 + subsample=2：保留帧 = 0,2,4,...（先抽帧，无抽稀）。"""
    make_episode(tmp_path, "task_a", "ep_001", n_frames=10)
    ds = Stage0MotionThinnedDataset(tmp_path, domain_id=0,
                                    camera="camera_front", motion_thresh=0.0,
                                    subsample=2, verbose=False)
    assert len(ds) == 5
    assert [ds.locate(i) for i in range(5)] == [(0, 2 * i) for i in range(5)]
    ds.close()


def test_subsample_then_thin(tmp_path):
    """先抽帧后抽稀：τ 作用在抽帧后的流上，帧号映射回原始。"""
    make_episode(tmp_path, "task_a", "ep_001", n_frames=9)
    import h5py
    h5 = next(tmp_path.glob("data/*/*/success_episodes/*/data/*.hdf5"))
    with h5py.File(h5, "a") as f:
        for k in ("puppet/arm_left_position_align",
                  "puppet/arm_right_position_align"):
            q = np.zeros((9, 8), np.float32)
            q[:, 0] = np.arange(9) * 0.05      # 每原始帧单关节 +0.05 rad
            f[f"{k}/data"][:] = q
    ds = Stage0MotionThinnedDataset(tmp_path, domain_id=0,
                                    camera="camera_front", motion_thresh=0.15,
                                    subsample=2, verbose=False)
    # 抽帧后流 = 帧 0,2,4,6,8（5 帧），相邻 Δ‖q‖ = 2×0.05×√2 ≈ 0.1414
    # 贪心：帧0 保留；累计 0.1414<0.15 跳帧2；0.2828≥0.15 保帧4；
    # 0.1414 跳帧6；0.2828 保帧8 ⇒ 原始帧号 [0, 4, 8]
    assert [ds.locate(i) for i in range(len(ds))] == [(0, 0), (0, 4), (0, 8)]
    ds.close()


# --------------------------------------------------------------------------- #
# latent 缓存与 clip 数据集
# --------------------------------------------------------------------------- #

class TinyEncoder(torch.nn.Module):
    """build_cache 用的假编码器：forward_tokens 返回确定性三通道。"""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, D)

    def forward_tokens(self, rgb):
        B = rgb.shape[0]
        h = self.lin(rgb.mean(dim=(2, 3)))     # (B, D)
        return {"cls": h,
                "registers": h[:, None].expand(B, R, D).contiguous(),
                "patch": h[:, None].expand(B, G * G, D).contiguous()}


def _build_tiny_cache(tmp_path, n_frames=12):
    make_episode(tmp_path / "real", "task_a", "ep_001", n_frames=n_frames,
                 seed=1)
    make_episode(tmp_path / "real", "task_a", "ep_002", n_frames=n_frames,
                 seed=2)
    ds = Stage0MotionThinnedDataset(tmp_path / "real", domain_id=0,
                                    camera="camera_front", motion_thresh=0.0,
                                    verbose=False)
    enc = TinyEncoder()
    recs = build_cache(ds, enc, tmp_path / "cache", "real", batch_size=4,
                       device="cpu", seed=0, val_frac=0.5, verbose=False)
    write_index(tmp_path / "cache", recs, meta={})
    ds.close()
    return tmp_path / "cache", recs


def test_cache_roundtrip(tmp_path):
    cache_dir, recs = _build_tiny_cache(tmp_path)
    assert len(recs) == 2 and all(r["n_frames"] == 12 for r in recs)
    ep = torch.load(cache_dir / recs[0]["file"], weights_only=True)
    assert ep["patch"].shape == (12, G * G, D)
    assert ep["registers"].shape == (12, R, D)
    assert ep["patch"].dtype == torch.bfloat16
    assert ep["rgb"].dtype == torch.uint8
    assert ep["action"].shape == (12, 16)
    # split 字段合法且确定性
    assert {r["split"] for r in recs} <= {"train", "val"}


def test_clip_dataset_windows(tmp_path):
    cache_dir, recs = _build_tiny_cache(tmp_path)
    split = recs[0]["split"]
    ds = LatentClipDataset(cache_dir, context=2, horizon=4, split=split)
    # 单集 12 帧、L=6 ⇒ 每集 7 个窗口；窗口不跨 episode（构造保证）
    assert len(ds) == 7 * sum(r["split"] == split for r in recs)
    item = ds[0]
    assert item["patch"].shape == (6, G * G, D)
    assert item["action"].shape == (6, 16)
    # frame_id 连续（stride=1）
    fi = item["frame_id"]
    assert torch.equal(fi[1:] - fi[:-1], torch.ones(5, dtype=torch.int64))


def test_clip_dataset_stride(tmp_path):
    cache_dir, recs = _build_tiny_cache(tmp_path)
    ds = LatentClipDataset(cache_dir, context=2, horizon=2,
                           split=recs[0]["split"], stride=2)
    item = ds[0]
    assert item["patch"].shape == (4, G * G, D)
    fi = item["frame_id"]
    assert torch.equal(fi[1:] - fi[:-1], 2 * torch.ones(3,
                                                        dtype=torch.int64))


# --------------------------------------------------------------------------- #
# 选项 A 梯度口径：信号梯度进 T，深度 stop-grad
# --------------------------------------------------------------------------- #

def test_option_a_gradient_flow():
    """joint_pos 挂 r̂：梯度必须到达 T；D 解码 ẑ.detach()：梯度不得到达 T。"""
    from sawvla.models import DepthDecoder
    adapter, T = make_t(depth=2)
    with torch.no_grad():
        T.head_reg.weight.normal_(0, 0.02)     # 打破恒等起步，否则 r̂≡r 无梯度
    heads = build_signals(
        {"latent": {"grid_size": G, "dim": D},
         "encoder": {"register_roles": ROLES},
         "signals": {"joint_pos": {"enabled": True, "weight": 1.0}}},
        dim=D, n_reg=R)
    Ddec = DepthDecoder(d=D, grid_size=G, out_size=G * 4, width1=16, width2=16)

    z, r, a, p = inputs(B=2, k=2)
    out = T.forward_with_registers(z, r, *adapter(a, p))
    ctx_hat = SignalContext(cls=torch.zeros(2, D),
                            registers=out["registers"][:, -1],
                            patch=out["patch"][:, -1], roles=ROLES)
    proprio = torch.randn(2, 16)
    sig_loss = sum(heads.losses(ctx_hat, {"proprio": proprio}).values())
    mu, _ = Ddec(out["patch"][:, -1].detach())  # 选项 A：深度对 T stop-grad
    depth_loss = mu.square().mean()
    (sig_loss + depth_loss).backward()

    t_grads = [q.grad for q in T.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in t_grads), \
        "信号损失梯度应到达 T（选项 A）"
    # 深度项若没 detach，conv 第一层就会给 T 额外梯度——验证 detach 生效：
    # 单独对深度项反传，T 必须零梯度
    for q in T.parameters():
        q.grad = None
    mu2, _ = Ddec(out["patch"][:, -1].detach())
    mu2.square().mean().backward()
    assert all(q.grad is None or q.grad.abs().sum() == 0
               for q in T.parameters()), "深度损失对 T 必须 stop-grad"
