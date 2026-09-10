"""下载预训练权重到 pretrained_models/(统一经 ModelScope,离线可复现)。

- teacher: depth-anything/Depth-Anything-V2-Large-hf
    Depth Anything V2-L 稠密深度教师,HF 格式(safetensors + config),
    由 transformers.DepthAnythingForDepthEstimation 原生加载,无需 vendor 官方源码。
- encoder: facebook/dinov2-base
    DINOv2-B/14,Stage 0 冻结 RGB 编码器(guideline v1 默认),HF 格式,
    由 transformers.Dinov2Model 原生加载。
- vlm: Qwen/Qwen3-VL-2B-Instruct
    Stage 3 VLA backbone,HF 格式,
    由 transformers.Qwen3VLForConditionalGeneration 原生加载。

用法:
    python scripts/download_pretrained.py                 # 全部下载
    python scripts/download_pretrained.py --only teacher  # 只下教师
    python scripts/download_pretrained.py --only encoder  # 只下编码器
    python scripts/download_pretrained.py --only vlm      # 只下 VLA backbone
    python scripts/download_pretrained.py --verify        # 下载并做加载自检
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "teacher": {
        "repo": "depth-anything/Depth-Anything-V2-Large-hf",
        "dir": "Depth-Anything-V2-Large-hf",
        "expect": ["config.json", "model.safetensors", "preprocessor_config.json"],
    },
    "encoder": {
        "repo": "facebook/dinov2-base",
        "dir": "dinov2-base",
        "expect": ["config.json", "model.safetensors", "preprocessor_config.json"],
    },
    "vlm": {
        "repo": "Qwen/Qwen3-VL-2B-Instruct",
        "dir": "Qwen3-VL-2B-Instruct",
        "expect": ["config.json", "model.safetensors", "tokenizer.json"],
    },
}


def download(name: str, root: Path) -> Path:
    from modelscope import snapshot_download

    spec = MODELS[name]
    local = root / spec["dir"]
    print(f"[{name}] {spec['repo']} -> {local}")
    snapshot_download(spec["repo"], local_dir=str(local))
    missing = [f for f in spec["expect"] if not (local / f).exists()]
    if missing:
        raise RuntimeError(f"[{name}] 下载不完整,缺少: {missing}")
    size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
    print(f"[{name}] OK, {size / 2**20:.0f} MiB")
    return local


def verify(name: str, local: Path) -> None:
    import torch
    from transformers import AutoImageProcessor

    if name == "teacher":
        from transformers import DepthAnythingForDepthEstimation
        model = DepthAnythingForDepthEstimation.from_pretrained(local)
    elif name == "encoder":
        from transformers import Dinov2Model
        model = Dinov2Model.from_pretrained(local)
    else:  # vlm
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        AutoProcessor.from_pretrained(local)
        model = Qwen3VLForConditionalGeneration.from_pretrained(local, dtype=torch.bfloat16)
    if name != "vlm":
        AutoImageProcessor.from_pretrained(local)
    n_params = sum(p.numel() for p in model.parameters())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    print(f"[{name}] 加载自检 OK: {n_params / 1e6:.0f}M params on {dev}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=[*MODELS], default=None, help="只下载某一个")
    ap.add_argument("--root", default=str(ROOT / "pretrained_models"))
    ap.add_argument("--verify", action="store_true", help="下载后用 transformers 做加载自检")
    args = ap.parse_args()

    names = [args.only] if args.only else list(MODELS)
    for name in names:
        local = download(name, Path(args.root))
        if args.verify:
            verify(name, local)
    return 0


if __name__ == "__main__":
    sys.exit(main())
