#!/usr/bin/env python
"""Stage 2 ckpt 扫描：对 outputs/stage2/ckpt_step*.pt + last.pt 逐个跑修正版
validate（分层多 episode 取样 + 独立探针），所有 ckpt 评**同一批** val 样本
（固定种子预收集），输出对照表 + JSON。

用法：
    conda activate style_agnostic_wam
    python scripts/eval_stage2_ckpts.py [--ckpt-dir outputs/stage2] [--max-batches 10]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_stage2 import build_clip_sets, validate  # noqa: E402
from sawvla.models import (ActionAdapter, DINOv2Encoder,  # noqa: E402
                           DepthDecoder, TransitionModel)
from sawvla.signals import build_signals  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=Path("outputs/stage2"))
    ap.add_argument("--max-batches", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-cfg", default="configs/model.yaml")
    ap.add_argument("--data-cfg", default="configs/data.yaml")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model_cfg = yaml.safe_load(open(args.model_cfg))
    data_cfg = yaml.safe_load(open(args.data_cfg))
    s1, s2 = model_cfg["stage1"], model_cfg["stage2"]
    C, K = int(s1["context_frames"]), int(s1["unroll_steps"])
    device = "cuda"

    _, _, val_per_dom = build_clip_sets(
        data_cfg, C, K, s2["clip_stride"], 0.02, args.seed, 0.1)
    # 预收集一次（固定种子 shuffle）：所有 ckpt 评同一批样本
    gen = torch.Generator().manual_seed(args.seed)
    val_loaders = [DataLoader(v, batch_size=args.batch, shuffle=True,
                              generator=gen, num_workers=args.workers,
                              pin_memory=True)
                   for v in val_per_dom]
    per_dom = max(1, args.max_batches // len(val_loaders))
    batch_sets = []
    for loader in val_loaders:
        bs = []
        for i, b in enumerate(loader):
            if i >= per_dom:
                break
            bs.append(b)
        batch_sets.append(bs)
    eps = [sorted({int(b["episode_id"][j]) for b in bs for j in
                   range(len(b["episode_id"]))}) for bs in batch_sets]
    print(f"val 样本：每域 {per_dom} 批 × {args.batch} clip；"
          f"episode 覆盖 real={len(eps[0])} 个 / sim={len(eps[1])} 个")

    # 模型骨架（逐 ckpt 装权重）
    tcfg = model_cfg["transition"]
    E = DINOv2Encoder(model_cfg["paths"]["encoder"]).to(device)
    T = TransitionModel(d=tcfg["d"], depth=tcfg["depth"], n_heads=tcfg["n_heads"],
                        grid_size=model_cfg["latent"]["grid_size"],
                        n_cond=tcfg["n_cond"], max_frames=tcfg["max_frames"],
                        n_reg=tcfg["n_reg"]).to(device)
    adapter = ActionAdapter(tcfg["action_dim"], tcfg["proprio_dim"],
                            d=tcfg["d"]).to(device)
    D = DepthDecoder(**model_cfg["depth_decoder"]).to(device)
    heads = build_signals(model_cfg, dim=model_cfg["latent"]["dim"],
                          n_reg=tcfg["n_reg"]).to(device)
    from sawvla.losses import DepthLoss
    loss_cfg = dict(model_cfg["loss"])
    ngs = loss_cfg.pop("n_grad_scales")
    loss_cfg["lambda_teacher"] = 0.0
    fns = {"depth": DepthLoss(n_grad_scales=ngs, **loss_cfg),
           "roles": model_cfg["encoder"]["register_roles"],
           "reg_dyn_weight": float(s1["reg_dyn_weight"]),
           "hsic": {}, "entropy": {},          # validate step=0 不触发清洗项
           "anchor_w": float(s2["anchor"]["distill_weight"])}
    models = {"E": E, "T": T, "adapter": adapter, "D": D, "heads": heads}

    ckpts = sorted(args.ckpt_dir.glob("ckpt_step*.pt"))
    last = args.ckpt_dir / "last.pt"
    if last.exists():
        ckpts.append(last)
    rows = []
    for p in ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)  # 自己的 ckpt
        E.load_state_dict(ck["E"]); T.load_state_dict(ck["T"])
        adapter.load_state_dict(ck["adapter"]); D.load_state_dict(ck["D"])
        heads.load_state_dict(ck["heads"])   # stage2 自身 ckpt 的键名是 heads
        va = validate(models, fns, batch_sets, C, K, device, s2,
                      max_batches=args.max_batches, probe_seed=args.seed)
        row = {"ckpt": p.name, "step": ck.get("step"),
               **{k: round(v, 5) for k, v in va.items()}}
        rows.append(row)
        print(f"{p.name:22s} step={ck.get('step'):>6} "
              f"probe_agnostic={va['probe_e/reg_agnostic']:.3f} "
              f"(prior={va['probe_prior']:.3f}) "
              f"patch={va['probe_e/patch']:.3f} dom={va['probe_e/reg_domain']:.3f} "
              f"depth={va['depth/true']:.3f} mse4={va['latent_mse/step_4']:.4f}",
              flush=True)
    out = args.out or (args.ckpt_dir / "eval_ckpts.json")
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
