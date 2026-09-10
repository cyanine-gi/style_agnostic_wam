"""转移模型 T 单测：逐步 token、block-causal 因果性、近恒等起步、RoPE、梯度通路。

因果泄漏测试（test_causal_leakage）对应 overall_tensor_flow.md §3.4 的硬性要求：
改动未来动作，过去的预测必须逐位不变。
"""

import torch

from sawvla.models import ActionAdapter, TransitionModel


def _make(d=384, grid=16, depth=2):
    torch.manual_seed(0)
    adapter = ActionAdapter(action_dim=14, proprio_dim=16, d=d)
    trans = TransitionModel(d=d, depth=depth, n_heads=6, grid_size=grid)
    return adapter, trans


def _inputs(B=2, k=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    z_seq = torch.randn(B, k, 256, 384, generator=g)
    actions = torch.randn(B, k, 14, generator=g)
    proprio = torch.randn(B, k, 16, generator=g)
    return z_seq, actions, proprio


def test_adapter_per_step_tokens():
    adapter, _ = _make()
    act_tok, prop_tok = adapter(torch.randn(2, 5, 14), torch.randn(2, 5, 16))
    assert act_tok.shape == (2, 5, 384) and prop_tok.shape == (2, 5, 384)
    # k 可变：同一 adapter 处理不同步数
    act_tok1, _ = adapter(torch.randn(2, 1, 14), torch.randn(2, 1, 16))
    assert act_tok1.shape == (2, 1, 384)
    # 逐步独立：第 j 个 token 只由第 j 步动作决定
    a = torch.zeros(1, 4, 14)
    a2 = a.clone(); a2[0, 3] = 1.0
    t1, _ = adapter(a, torch.zeros(1, 4, 16))
    t2, _ = adapter(a2, torch.zeros(1, 4, 16))
    assert torch.equal(t1[0, :3], t2[0, :3]) and not torch.equal(t1[0, 3], t2[0, 3])


def test_forward_shape():
    adapter, trans = _make()
    z_seq, actions, proprio = _inputs(k=3)
    z_hat = trans(z_seq, *adapter(actions, proprio))
    assert z_hat.shape == z_seq.shape


def test_near_identity_at_init():
    # 输出头零初始化 ⇒ 初始 ᑮ ≡ z（残差 delta 设计）
    adapter, trans = _make()
    z_seq, actions, proprio = _inputs()
    z_hat = trans(z_seq, *adapter(actions, proprio))
    assert torch.allclose(z_hat, z_seq, atol=1e-5)


def test_causal_leakage():
    # §3.4：两个仅未来动作不同的输入，过去步的预测必须逐位相同
    adapter, trans = _make(depth=3)
    with torch.no_grad():  # head 零初始化时输出恒等于输入，无法检验掩码 → 先打破
        trans.head.weight.normal_(0, 0.02)
    z_seq, actions, proprio = _inputs(B=1, k=4)
    actions2 = actions.clone()
    actions2[0, 3] += 5.0                      # 只改第 3 步动作（未来）
    out1 = trans(z_seq, *adapter(actions, proprio))
    out2 = trans(z_seq, *adapter(actions2, proprio))
    # 第 0/1/2 步预测（只能见 a_{0..2}）必须逐位不变
    assert torch.equal(out1[:, :3], out2[:, :3])
    # 第 3 步预测允许变（掩码没把动作通道彻底隔断的 sanity 检查）
    assert not torch.allclose(out1[:, 3], out2[:, 3])


def test_action_changes_output():
    adapter, trans = _make()
    z_seq, actions, proprio = _inputs(B=1, k=2)
    a2 = actions + torch.randn_like(actions)
    opt = torch.optim.SGD(list(trans.parameters()) + list(adapter.parameters()), lr=1.0)
    trans(z_seq, *adapter(actions, proprio)).sum().backward()
    opt.step()
    out1 = trans(z_seq, *adapter(actions, proprio))
    out2 = trans(z_seq, *adapter(a2, proprio))
    assert not torch.allclose(out1, out2)


def test_gradient_flows_to_input_and_cond():
    adapter, trans = _make()
    with torch.no_grad():
        # head 零初始化时输出恒等于 z，cond 梯度必为 0（恒等起步的设计行为）；
        # 给 head 非零权重后再验证梯度通路
        trans.head.weight.normal_(0, 0.02)
    z_seq, actions, proprio = _inputs(B=1, k=2)
    z_seq.requires_grad_(True)
    act_tok, prop_tok = adapter(actions, proprio)
    act_tok.retain_grad()                      # 非叶子张量，需显式 retain
    trans(z_seq, act_tok, prop_tok).sum().backward()
    assert z_seq.grad is not None and z_seq.grad.abs().sum() > 0
    assert act_tok.grad is not None and act_tok.grad.abs().sum() > 0


def test_rope_is_position_aware():
    # 2D-RoPE：同一 patch 内容放在不同网格位置，主干隐状态应不同
    adapter, trans = _make()
    z = torch.zeros(1, 1, 256, 384)
    z[0, 0, 0] = 1.0                           # 内容放左上角
    z2 = torch.zeros(1, 1, 256, 384)
    z2[0, 0, 255] = 1.0                        # 同内容放右下角
    cond = torch.randn(1, 1, 14), torch.randn(1, 1, 16)
    act_tok, prop_tok = adapter(*cond)
    x = torch.cat([z, prop_tok[:, :, None], act_tok[:, :, None]], dim=2)
    x2 = torch.cat([z2, prop_tok[:, :, None], act_tok[:, :, None]], dim=2)
    emb = trans.frame_embed(torch.tensor([0]))[None, :, None, :]
    x = (x + emb).reshape(1, trans.n_block, -1)
    x2 = (x2 + emb).reshape(1, trans.n_block, -1)
    for blk in trans.blocks:
        x, x2 = blk(x, 1, None), blk(x2, 1, None)
    assert not torch.allclose(x, x2)


def test_frame_embed_changes_output():
    adapter, trans = _make()
    with torch.no_grad():
        trans.head.weight.normal_(0, 0.02)
        trans.frame_embed.weight[1] += 1.0
    z_seq, actions, proprio = _inputs(B=1, k=1)
    out0 = trans(z_seq, *adapter(actions, proprio), frame_offset=0)
    out1 = trans(z_seq, *adapter(actions, proprio), frame_offset=1)
    assert not torch.allclose(out0, out1)


def test_unroll_shape_and_matches_single_steps():
    adapter, trans = _make()
    z_seq, actions, proprio = _inputs(B=2, k=4)
    act_tok, prop_tok = adapter(actions, proprio)
    preds = trans.unroll(z_seq[:, 0], act_tok, prop_tok)
    assert preds.shape == z_seq.shape
    # rollout 第 1 步应与 teacher-forced 第 1 步一致（同输入同参数）
    tf = trans(z_seq[:, :1], act_tok[:, :1], prop_tok[:, :1])
    assert torch.allclose(preds[:, 0], tf[:, 0], atol=1e-5)


def test_param_count():
    trans = TransitionModel(d=384, depth=8, n_heads=6)
    n = sum(p.numel() for p in trans.parameters()) / 1e6
    assert 10 < n < 25                         # 默认 ≈20M，上限 100M 远未到
