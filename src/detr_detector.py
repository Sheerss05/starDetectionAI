"""
detr_detector.py
─────────────────
Step 4 — DETR (Detection Transformer) Constellation Detector

Uses a HuggingFace DETR model fine-tuned on constellation imagery.
DETR analyses global image context through attention mechanisms, making it
especially effective for dense star fields where YOLO may produce duplicates.

Architecture
------------
  facebook/detr-resnet-50  (pretrained backbone)
  → fine-tuned with constellation bounding-box annotations
  → 88 constellation classes + background

The ``DetrDetector.detect()`` method returns the same ``Detection`` dataclass
used by the YOLO detector so both outputs can be fused uniformly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image as PILImage

from src.yolo_detector import Detection   # shared Detection dataclass

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# DETR Detector
# ──────────────────────────────────────────────────────────────────────────────

class DetrDetector:
    """
    HuggingFace DETR-based constellation detector.

    Parameters
    ----------
    model_weights : str | Path
        Path to fine-tuned DETR checkpoint directory / .pt file.
    pretrained_base : str
        HuggingFace model ID used as backbone when weights are absent.
    img_size : int
        Longest edge size fed to the DETR processor.
    conf_threshold : float
        Minimum logit-converted probability to report a detection.
    num_queries : int
        Number of object queries (DETR hyperparameter).
    device : str
        "cuda" | "cpu".
    num_classes : int
        Number of constellation classes (excluding background).
    class_names : list[str] | None
        Ordered class labels; index must align with model head outputs.
    annotation_file : str | Path | None
        Path to a COCO-format annotation JSON. When provided, class names are
        read from the ``categories`` field and override ``class_names``.
    """

    def __init__(
        self,
        model_weights: str | Path = "models/detr/constellation_detr.pt",
        pretrained_base: str = "facebook/detr-resnet-50",
        img_size: int = 800,
        conf_threshold: float = 0.30,
        num_queries: int = 100,
        device: str = "cuda",
        num_classes: int = 88,
        class_names: Optional[List[str]] = None,
        annotation_file: Optional[str | Path] = None,
    ) -> None:
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.num_queries = num_queries
        self.device = self._resolve_device(device)
        self.num_classes = num_classes
        # Resolve class names: annotation_file > explicit class_names > numeric fallback
        ann_names = self._load_class_names_from_annotation(annotation_file) if annotation_file else []
        self.class_names = ann_names or class_names or [str(i) for i in range(num_classes)]
        self.model, self.processor = self._load_model(model_weights, pretrained_base)

    # ── inference ─────────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run DETR detection on a preprocessed image.

        Parameters
        ----------
        image : np.ndarray
            Float32 (H, W, 3) in [0, 1]  OR  uint8 (H, W, 3).

        Returns
        -------
        List[Detection]
        """
        if self.model is None or self.processor is None:
            logger.warning("DETR model not loaded — returning empty detections.")
            return []

        pil_img = self._to_pil(image)
        h, w = image.shape[:2]

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process: rescale boxes back to original image coordinates
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.conf_threshold,
        )

        detections: List[Detection] = []
        for result in results:
            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            boxes = result["boxes"].cpu().numpy()   # xyxy

            for score, cls_id, box in zip(scores, labels, boxes):
                label = (
                    self.class_names[int(cls_id)]
                    if int(cls_id) < len(self.class_names)
                    else str(cls_id)
                )
                detections.append(Detection(
                    label=label,
                    bbox=box.tolist(),
                    confidence=float(score),
                    source="detr",
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
            from transformers import DetrForObjectDetection, DetrImageProcessor
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers")
            return None, None

        processor = DetrImageProcessor.from_pretrained(base)

        weights_path = Path(weights_path)
        companion = weights_path.parent / "detr_checkpoint.pt"

        state = self._resolve_state_dict(weights_path, companion)

        # Auto-detect num_classes from the classifier head stored in the checkpoint.
        # The head bias has shape [num_classes + 1] (background is the last slot).
        if state is not None and "class_labels_classifier.bias" in state:
            detected = int(state["class_labels_classifier.bias"].shape[0]) - 1
            if detected != self.num_classes:
                logger.info(
                    f"Checkpoint has {detected} classes (configured: {self.num_classes}). "
                    "Updating num_classes to match checkpoint."
                )
                self.num_classes = detected
                # Only expand/shrink numeric placeholder names, not user-supplied names
                if all(n.isdigit() for n in self.class_names):
                    self.class_names = [str(i) for i in range(self.num_classes)]

        model = DetrForObjectDetection.from_pretrained(
            base,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )

        if state is not None:
            result = model.load_state_dict(state, strict=False)
            logger.info(
                f"DETR weights loaded — missing: {len(result.missing_keys)}, "
                f"unexpected: {len(result.unexpected_keys)}"
            )
        else:
            logger.warning("No DETR weights found — base model loaded (predictions will be meaningless).")

        model.eval()
        model.to(self.device)
        return model, processor

    def _resolve_state_dict(self, weights_path: Path, companion: Path):
        """Return the best available state dict, preferring the fine-tuned checkpoint."""

        def _load(p: Path):
            raw = torch.load(p, map_location=self.device, weights_only=True)
            # Unwrap training-checkpoint envelope {epoch, model, optimizer, …}
            if isinstance(raw, dict) and "epoch" in raw and "model" in raw:
                return raw["model"]
            return raw

        def _num_classes_in(state):
            if "class_labels_classifier.bias" in state:
                return int(state["class_labels_classifier.bias"].shape[0]) - 1
            return None

        primary_state = None
        if weights_path.exists():
            try:
                primary_state = _load(weights_path)
                logger.info(f"Loaded DETR weights from {weights_path}")
            except Exception as exc:
                logger.warning(f"Failed to load {weights_path}: {exc}")

        companion_state = None
        if companion.exists():
            try:
                companion_state = _load(companion)
                logger.info(f"Loaded DETR checkpoint from {companion}")
            except Exception as exc:
                logger.warning(f"Failed to load {companion}: {exc}")

        # Prefer the checkpoint whose num_classes matches self.num_classes.
        # If neither matches, prefer the companion (epoch-based training checkpoint)
        # over the primary (which may still hold pretrained-COCO weights).
        if primary_state is not None:
            primary_nc = _num_classes_in(primary_state)
            if primary_nc == self.num_classes:
                return primary_state          # exact match — use primary
            # Primary has wrong class count. Try companion.
            if companion_state is not None:
                companion_nc = _num_classes_in(companion_state)
                if companion_nc == self.num_classes or companion_nc is not None:
                    logger.info(
                        f"Primary weights have {primary_nc} classes (expected {self.num_classes}); "
                        f"using companion checkpoint ({companion_nc} classes)."
                    )
                    return companion_state
            # No better option — return primary (auto-detect will correct num_classes)
            return primary_state

        return companion_state  # primary missing; use companion if it exists

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
    def _to_pil(image: np.ndarray) -> PILImage.Image:
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return PILImage.fromarray(image)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def detect_detr(
    image: np.ndarray,
    cfg: dict | None = None,
    class_names: Optional[List[str]] = None,
) -> List[Detection]:
    cfg = cfg or {}
    detector = DetrDetector(
        model_weights=cfg.get("model_weights", "models/detr/constellation_detr.pt"),
        pretrained_base=cfg.get("pretrained_base", "facebook/detr-resnet-50"),
        img_size=cfg.get("img_size", 800),
        conf_threshold=cfg.get("conf_threshold", 0.30),
        num_queries=cfg.get("num_queries", 100),
        device=cfg.get("device", "cuda"),
        num_classes=cfg.get("num_classes", 88),
        class_names=class_names,
    )
    return detector.detect(image)
