"""
yolo_detector.py
─────────────────
Step 3 — YOLO-based Constellation Detection

Wraps Ultralytics YOLOv8 for fast, region-level constellation detection.
During inference each detected region produces:
  • constellation label
  • bounding box  [x1, y1, x2, y2]  (pixel coordinates)
  • confidence score

The weights at ``model_weights`` are expected to be a fine-tuned YOLOv8
checkpoint trained on annotated constellation images.  If the checkpoint is
absent, the model falls back to the pretrained base for structural inspection
only (no meaningful predictions).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Detection result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single constellation detection from any detector."""
    label: str
    bbox: List[float]          # [x1, y1, x2, y2] in pixel space
    confidence: float
    source: str = "unknown"    # "yolo" | "detr" | "fused"
    verified_by_gnn: bool = False
    gnn_score: float = 0.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def centre(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def to_dict(self) -> dict:
        return {
            "constellation_name": self.label,
            "bounding_box": self.bbox,
            "confidence_score": round(self.confidence, 4),
            "verified_by_GNN": self.verified_by_gnn,
            "gnn_score": round(self.gnn_score, 4),
            "source": self.source,
        }

    def __repr__(self) -> str:
        return (
            f"Detection(label={self.label!r}, conf={self.confidence:.2f}, "
            f"bbox={[round(v, 1) for v in self.bbox]}, src={self.source})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# YOLO Detector
# ──────────────────────────────────────────────────────────────────────────────

class YOLODetector:
    """
    YOLOv8-based constellation detector.

    Parameters
    ----------
    model_weights : str | Path
        Path to fine-tuned .pt weights.
    pretrained_base : str
        Ultralytics model name used if weights file is missing.
    img_size : int
        Inference image size.
    conf_threshold : float
        Minimum detection confidence to report.
    iou_threshold : float
        NMS IoU threshold.
    device : str
        "cuda" | "cpu" | "mps".
    num_classes : int
        Number of constellation classes.
    class_names : list[str] | None
        Optional ordered list of class names.
    """

    def __init__(
        self,
        model_weights: str | Path = "models/yolo/constellation_yolo.pt",
        pretrained_base: str = "yolov8m.pt",
        img_size: int = 640,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        num_classes: int = 88,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = self._resolve_device(device)
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.model = self._load_model(model_weights, pretrained_base)

    # ── inference ─────────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run YOLO detection on a preprocessed image.

        Parameters
        ----------
        image : np.ndarray
            Float32 (H, W, 3) in [0, 1]  OR  uint8 (H, W, 3) in [0, 255].

        Returns
        -------
        List[Detection]
        """
        if self.model is None:
            logger.warning("YOLO model not loaded — returning empty detections.")
            return []

        # Ultralytics expects uint8 or float32; ensure uint8 for stability
        img_u8 = self._to_uint8(image)

        results = self.model.predict(
            source=img_u8,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                label = (
                    self.class_names[cls_id]
                    if cls_id < len(self.class_names)
                    else str(cls_id)
                )
                detections.append(Detection(
                    label=label,
                    bbox=xyxy,
                    confidence=conf,
                    source="yolo",
                ))

        return detections

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self, weights_path: str | Path, base: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics is not installed. Install with: pip install ultralytics")
            return None

        weights_path = Path(weights_path)
        if weights_path.exists():
            logger.info(f"Loading YOLO weights from {weights_path}")
            model = YOLO(str(weights_path))
        else:
            logger.warning(
                f"YOLO weights not found at {weights_path}. "
                f"Loading base model '{base}' — predictions will be meaningless."
            )
            model = YOLO(base)

        # Use class names embedded in the checkpoint — these are the exact names
        # used during training and must override any config-supplied names.
        if hasattr(model, 'names') and model.names:
            embedded = model.names
            if isinstance(embedded, dict):
                self.class_names = [embedded[i] for i in sorted(embedded.keys())]
            else:
                self.class_names = list(embedded)
            logger.info(f"Using {len(self.class_names)} class names from checkpoint")

        model.to(self.device)
        return model

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable — using CPU.")
            return "cpu"
        return device

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def detect_yolo(
    image: np.ndarray,
    cfg: dict | None = None,
    class_names: Optional[List[str]] = None,
) -> List[Detection]:
    cfg = cfg or {}
    detector = YOLODetector(
        model_weights=cfg.get("model_weights", "models/yolo/constellation_yolo.pt"),
        pretrained_base=cfg.get("pretrained_base", "yolov8m.pt"),
        img_size=cfg.get("img_size", 640),
        conf_threshold=cfg.get("conf_threshold", 0.30),
        iou_threshold=cfg.get("iou_threshold", 0.45),
        device=cfg.get("device", "cuda"),
        num_classes=cfg.get("num_classes", 88),
        class_names=class_names,
    )
    return detector.detect(image)
