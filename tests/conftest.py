import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ENCODER_PATH = ROOT / "pretrained_models" / "dinov2-small-reg"
