"""三个预训练模型的统一推理冒烟测试:教师(DA-V2-L)、编码器(DINOv2-B)、VLM(Qwen3-VL-2B)。

模型路径与选择全部读自 configs/model.yaml(guideline §11.2)。在同一条 real
episode 的同一帧上测三个模型,输出:

- teacher: 稠密深度推理,与 GT 深度并排存图(重点:GT 洞内教师是否给出合理
  值),并测 batch8@518 吞吐,用于校准全量教师离线推理的时长估算;
- encoder: 验证 patch token 形状(patch14 -> 224px 输入应 1+16x16=257 tokens
  @ 768 维),这是 Stage 0 attention pooling 的输入;
- vlm: 图像描述一轮,验证 图像编码->对齐->生成 全链路。

用法: python scripts/test_pretrained_models.py [--hdf5 PATH] [--frame 500]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "introspect" / "pretrained_models_test.png"

DEFAULT_HDF5 = ("data/RoboMIND2.0-Tienkung/data/tienkung/tidy_desktop/"
                "success_episodes/0115_153224/data/trajectory.hdf5")


def load_frame(path: str, idx: int):
    with h5py.File(path, "r") as f:
        cbuf = f["camera_observations/color_images/camera_top"][idx]
        dbuf = f["camera_observations/depth_images/camera_top"][idx]
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(cbuf, np.uint8), cv2.IMREAD_COLOR),
                       cv2.COLOR_BGR2RGB)
    gt = cv2.imdecode(np.frombuffer(dbuf, np.uint8), cv2.IMREAD_UNCHANGED)
    return rgb, gt


def colorize(depth01: np.ndarray, holes: np.ndarray | None = None) -> np.ndarray:
    vis = cv2.applyColorMap((depth01 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    if holes is not None:
        vis[holes] = (0, 0, 255)  # 洞标红
    return vis


def release():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_teacher(cfg, rgb, gt, device):
    from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

    path = ROOT / cfg["paths"]["teacher"]
    size = cfg["teacher"]["input_size"]
    proc = AutoImageProcessor.from_pretrained(path)
    model = DepthAnythingForDepthEstimation.from_pretrained(path).to(device).eval()

    with torch.inference_mode():
        inputs = proc(images=Image.fromarray(rgb), return_tensors="pt").to(device)
        pred = model(**inputs).predicted_depth  # (1, h, w) 相对深度,值大=近
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=gt.shape, mode="bicubic", align_corners=False)[0, 0]
        pred = pred.cpu().numpy()

    # 推理耗时:batch 8 @ 518px,20 次(前 5 次热身)
    batch = inputs["pixel_values"].repeat(8, 1, 1, 1)
    with torch.inference_mode():
        for _ in range(5):
            model(pixel_values=batch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            model(pixel_values=batch)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 20
    fps = 8 / dt
    vram = torch.cuda.max_memory_allocated() / 2**30

    # 洞区域教师覆盖检查
    holes = gt == 0
    hole_std = pred[holes].std() if holes.any() else float("nan")
    print(f"[teacher] batch8@{size}: {dt*1000:.0f} ms/iter = {fps:.1f} fps, "
          f"peak VRAM {vram:.2f} GiB")
    n_frames = 3.4e6
    print(f"[teacher] 全量 ~3.4M 帧估算: {n_frames / fps / 3600:.1f} h "
          f"(隔 3 帧推理约 {n_frames / 3 / fps / 3600:.1f} h)")
    print(f"[teacher] GT 洞占比 {holes.mean()*100:.1f}%, 洞内教师预测 std={hole_std:.4f} "
          f"(>0 说明教师在洞内给出了有变化的合理值,不是常数)")

    # 存图: RGB | GT(洞标红) | 教师稠密
    valid = ~holes
    gt01 = np.zeros_like(gt, np.float32)
    gt01[valid] = (gt[valid] - gt[valid].min()) / max(np.ptp(gt[valid]), 1)
    p01 = (pred - pred.min()) / max(np.ptp(pred), 1)
    h = 400
    row = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colorize(gt01, holes), colorize(p01)]
    row = [cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h)) for x in row]
    cv2.imwrite(str(OUT), np.hstack(row))
    print(f"[teacher] 对比图 -> {OUT}")
    del model
    release()


def test_encoder(cfg, rgb, device):
    from transformers import AutoImageProcessor, Dinov2Model

    path = ROOT / cfg["paths"]["encoder"]
    proc = AutoImageProcessor.from_pretrained(path)
    model = Dinov2Model.from_pretrained(path).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    with torch.inference_mode():
        inputs = proc(images=Image.fromarray(rgb), return_tensors="pt").to(device)
        out = model(**inputs)
    B, N, D = out.last_hidden_state.shape
    patch = cfg["encoder"]["patch"]
    h, w = inputs["pixel_values"].shape[-2:]
    expect = 1 + (h // patch) * (w // patch)
    ok = "OK" if N == expect else "MISMATCH"
    print(f"[encoder] DINOv2-B {n_params/1e6:.0f}M params, tokens {tuple(out.last_hidden_state.shape)} "
          f"(输入 {h}x{w}, patch{patch} -> 预期 {expect} tokens, {ok})")
    print(f"[encoder] peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    del model
    release()


def test_vlm(cfg, rgb, device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    path = ROOT / cfg["paths"]["vlm"]
    processor = AutoProcessor.from_pretrained(path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, device_map=device,
        attn_implementation=cfg["vlm"]["attn_implementation"],
    ).eval()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": Image.fromarray(rgb)},
            {"type": "text", "text": "Describe the robot and objects on the table in one sentence."},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128)
    text = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(f"[vlm] reply: {text}")
    print(f"[vlm] peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    del model
    release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default=str(ROOT / DEFAULT_HDF5))
    ap.add_argument("--frame", type=int, default=500)
    ap.add_argument("--cfg", default=str(ROOT / "configs" / "model.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rgb, gt = load_frame(args.hdf5, args.frame)
    print(f"frame: {args.hdf5} [{args.frame}], rgb {rgb.shape}, gt {gt.shape} {gt.dtype}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    test_teacher(cfg, rgb, gt, device)
    test_encoder(cfg, rgb, device)
    test_vlm(cfg, rgb, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
