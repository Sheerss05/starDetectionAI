"""
train_yolo.py
──────────────
Fine-tune a YOLOv8 model on the constellation dataset.

Expected dataset layout (Ultralytics YOLOv8 format):
  data/constellation_dataset/
    images/
      train/  *.jpg / *.png
      val/    *.jpg / *.png
    labels/
      train/  *.txt   (YOLO format: class cx cy w h  — normalised [0,1])
      val/    *.txt
    dataset.yaml

Run
───
  python training/train_yolo.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Training entry point
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed.  Run: pip install ultralytics")

    yolo_cfg   = cfg.get("yolo", {})
    train_cfg  = cfg.get("train_yolo", {})

    # ── Resume detection ──────────────────────────────────────────────────────
    # Use last.pt as starting weights if available (works cross-machine).
    # resume=True is intentionally avoided because it relies on absolute paths
    # stored inside the checkpoint that are only valid on the original machine.
    checkpoint = Path("models/yolo/constellation_run/weights/last.pt")
    if checkpoint.exists():
        logger.info(f"Loading checkpoint weights from: {checkpoint}")
        weights = str(checkpoint)
    else:
        weights = yolo_cfg.get("pretrained_base", "yolov8m.pt")
        logger.info(f"No checkpoint found. Loading base model: {weights}")

    model = YOLO(weights)

    # ── Build / validate dataset.yaml ────────────────────────────────────────
    data_yaml = Path(train_cfg.get("data_yaml", "data/constellation_dataset/dataset.yaml"))
    if not data_yaml.exists():
        logger.warning(
            f"dataset.yaml not found at {data_yaml}.  "
            "Creating a stub — please fill in paths before training."
        )
        _create_stub_dataset_yaml(data_yaml, num_classes=yolo_cfg.get("num_classes", 88))

    # ── Training ──────────────────────────────────────────────────────────────
    results = model.train(
        data=str(data_yaml),
        epochs=train_cfg.get("epochs", 100),
        imgsz=train_cfg.get("img_size", 640),
        batch=train_cfg.get("batch_size", 16),
        lr0=train_cfg.get("lr0", 0.01),
        augment=train_cfg.get("augment", True),
        device=yolo_cfg.get("device", "cuda"),
        project="models/yolo",
        name="constellation_run",
        exist_ok=True,
    )

    # ── Save final weights ────────────────────────────────────────────────────
    out_weights = Path(yolo_cfg.get("model_weights", "models/yolo/constellation_yolo.pt"))
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_weights))
    logger.info(f"Training complete.  Weights saved to {out_weights}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation configuration helper
# ──────────────────────────────────────────────────────────────────────────────

AUGMENTATION_ARGS = dict(
    degrees=180.0,          # full rotation support
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.5,
    hsv_v=0.4,              # brightness variation
    mosaic=0.5,
    mixup=0.2,
)


# ──────────────────────────────────────────────────────────────────────────────
# Stub dataset.yaml
# ──────────────────────────────────────────────────────────────────────────────

def _create_stub_dataset_yaml(path: Path, num_classes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stub = {
        "path": str(path.parent.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    num_classes,
        "names": [f"constellation_{i}" for i in range(num_classes)],
    }
    with open(path, "w") as f:
        yaml.dump(stub, f, default_flow_style=False)
    logger.info(f"Stub dataset.yaml written to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLO on constellation dataset")
    p.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
