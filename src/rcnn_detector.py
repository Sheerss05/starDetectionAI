"""
rcnn_detector.py
─────────────────
Step 5 — Faster R-CNN Constellation Detector

Uses torchvision's Faster R-CNN (ResNet-50-FPN backbone) fine-tuned on
constellation imagery.  Complements YOLO (fast single-pass feed-forward) and
DETR (global attention) with strong region-proposal detection, making it
effective for partially overlapping constellations and variable-scale fields.

The ``RCNNDetector.detect()`` returns the same ``Detection`` dataclass
used by YOLO and DETR so all three outputs can be fused uniformly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from src.yolo_detector import Detection   # shared Detection dataclass

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# RCNN Detector
# ──────────────────────────────────────────────────────────────────────────────

class RCNNDetector:
    """
    Faster R-CNN constellation detector (torchvision).

    Parameters
    ----------
    model_weights : str | Path
        Path to fine-tuned .pt weights file.
    pretrained_base : str
        Torchvision model variant used as backbone when fine-tuned weights are
        absent.  Supported values: "fasterrcnn_resnet50_fpn" (default),
        "fasterrcnn_resnet50_fpn_v2".
    img_size : int
        Longest-edge resize fed to the model (torchvision handles scaling).
    conf_threshold : float
        Minimum predicted score to keep a detection.
    iou_threshold : float
        NMS IoU threshold.
    device : str
        "cuda" | "cpu".
    num_classes : int
        Number of constellation classes (background is added automatically).
    class_names : list[str] | None
        Ordered class labels; index must align with model head outputs.
    annotation_file : str | Path | None
        Path to a COCO-format annotation JSON.  When provided, class names are
        read from the ``categories`` field (sorted by id) and override
        ``class_names``.
    """

    def __init__(
        self,
        model_weights: str | Path = "models/rcnn/constellation_rcnn.pt",
        pretrained_base: str = "fasterrcnn_resnet50_fpn",
        img_size: int = 800,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        num_classes: int = 88,
        class_names: Optional[List[str]] = None,
        annotation_file: Optional[str | Path] = None,
    ) -> None:
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = self._resolve_device(device)
        self.num_classes = num_classes
        ann_names = self._load_class_names_from_annotation(annotation_file) if annotation_file else []
        self.class_names = ann_names or class_names or [str(i) for i in range(num_classes)]
        self.model = self._load_model(model_weights, pretrained_base)

    # ── inference ─────────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run Faster R-CNN detection on a preprocessed image.

        Parameters
        ----------
        image : np.ndarray
            uint8 (H, W, 3) RGB  OR  float32 (H, W, 3) in [0, 1].

        Returns
        -------
        List[Detection]
        """
        if self.model is None:
            logger.warning("RCNN model not loaded — returning empty detections.")
            return []

        tensor = self._to_tensor(image).to(self.device)

        with torch.no_grad():
            outputs = self.model([tensor])

        # TorchScript FasterRCNN returns Tuple[Dict[losses], List[Dict[detections]]]
        # Regular (eager) FasterRCNN returns List[Dict[detections]] directly.
        if (
            isinstance(outputs, (tuple, list))
            and len(outputs) == 2
            and isinstance(outputs[0], dict)
            and isinstance(outputs[1], (list, tuple))
        ):
            detection_outputs = outputs[1]   # scripted model
        else:
            detection_outputs = outputs      # eager model

        detections: List[Detection] = []
        for output in detection_outputs:
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            boxes  = output["boxes"].cpu().numpy()   # xyxy

            for score, cls_id, box in zip(scores, labels, boxes):
                if float(score) < self.conf_threshold:
                    continue
                # Faster R-CNN label indices start at 1 (0 = background)
                idx = int(cls_id) - 1
                label = (
                    self.class_names[idx]
                    if 0 <= idx < len(self.class_names)
                    else str(cls_id)
                )
                detections.append(Detection(
                    label=label,
                    bbox=box.tolist(),
                    confidence=float(score),
                    source="rcnn",
                ))

        # Keep at most 1 detection per constellation label (highest confidence first)
        label_counts: dict = {}
        filtered: List[Detection] = []
        for d in sorted(detections, key=lambda x: x.confidence, reverse=True):
            count = label_counts.get(d.label, 0)
            if count < 1:
                filtered.append(d)
                label_counts[d.label] = count + 1
        return filtered

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self, weights_path: str | Path, base: str):
        try:
            import torchvision  # must be imported first to register torchvision C++ ops
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn,
                fasterrcnn_resnet50_fpn_v2,
            )
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        except ImportError:
            logger.error("torchvision not installed. Install with: pip install torchvision")
            return None

        weights_path = Path(weights_path)
        if not weights_path.exists():
            logger.warning(
                f"RCNN weights not found at {weights_path}. "
                "Using un-trained weights — predictions will be meaningless."
            )
            # Fall through to build a random-weight model
        else:
            # ── Try TorchScript archive first (.pt saved via torch.jit.script) ──
            try:
                model = torch.jit.load(str(weights_path), map_location=self.device)
                model.eval()
                logger.info(f"Loaded RCNN TorchScript model from {weights_path}")
                return model
            except Exception as jit_exc:
                logger.debug(f"TorchScript load failed ({jit_exc}), trying state-dict load …")

            # ── Fall back: state-dict (.pth / checkpoint .pt) ──────────────────
            # Background is class index 0 → total = num_classes + 1
            num_classes_with_bg = self.num_classes + 1

            if base == "fasterrcnn_resnet50_fpn_v2":
                model = fasterrcnn_resnet50_fpn_v2(weights=None)
            else:
                model = fasterrcnn_resnet50_fpn(weights=None)

            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)

            try:
                state = torch.load(weights_path, map_location=self.device, weights_only=False)
                # Unwrap training-checkpoint envelope if present
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif isinstance(state, dict) and "model" in state:
                    state = state["model"]
                result = model.load_state_dict(state, strict=False)
                logger.info(
                    f"RCNN weights loaded — missing: {len(result.missing_keys)}, "
                    f"unexpected: {len(result.unexpected_keys)}"
                )
            except Exception as exc:
                logger.warning(f"Failed to load RCNN weights: {exc}. Using random weights.")

            model.eval()
            model.to(self.device)
            return model

        # No weights file — build untrained model
        num_classes_with_bg = self.num_classes + 1
        if base == "fasterrcnn_resnet50_fpn_v2":
            model = fasterrcnn_resnet50_fpn_v2(weights=None)
        else:
            model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
        model.eval()
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
    def _load_class_names_from_annotation(annotation_file: str | Path) -> List[str]:
        """Return ordered class names from a COCO-format annotation JSON."""
        path = Path(annotation_file)
        if not path.exists():
            logger.warning(f"Annotation file not found: {path}")
            return []
        try:
            with open(path) as f:
                coco = json.load(f)
            categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
            names = [c["name"] for c in categories]
            logger.info(f"Loaded {len(names)} class names from {path}")
            return names
        except Exception as exc:
            logger.warning(f"Failed to read class names from {path}: {exc}")
            return []

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        """Convert numpy HWC image to CHW float32 tensor in [0, 1]."""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = np.clip(image, 0.0, 1.0).astype(np.float32)
        # HWC → CHW
        return torch.from_numpy(image.transpose(2, 0, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def detect_rcnn(
    image: np.ndarray,
    cfg: dict | None = None,
    class_names: Optional[List[str]] = None,
) -> List[Detection]:
    cfg = cfg or {}
    detector = RCNNDetector(
        model_weights=cfg.get("model_weights", "models/rcnn/constellation_rcnn.pt"),
        pretrained_base=cfg.get("pretrained_base", "fasterrcnn_resnet50_fpn"),
        img_size=cfg.get("img_size", 800),
        conf_threshold=cfg.get("conf_threshold", 0.30),
        iou_threshold=cfg.get("iou_threshold", 0.45),
        device=cfg.get("device", "cuda"),
        num_classes=cfg.get("num_classes", 88),
        class_names=class_names,
    )
    return detector.detect(image)
