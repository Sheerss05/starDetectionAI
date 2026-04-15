"""
fusion.py
──────────
Step 6 — Multi-Model Result Fusion

Merges detections from YOLO, DETR, and RCNN into a final deduplicated,
verified constellation list.

Fusion Rules
────────────
A detection is accepted when ANY of the following conditions hold:

  1. Model Agreement  — ≥ min_model_agreement detectors produce an
                        overlapping bbox (IoU ≥ iou_merge_threshold)
                        for the same constellation label.

  2. High Confidence  — A single detector reaches a very high confidence
                        (> 0.90), treating it as reliable enough to accept
                        without cross-detector confirmation.

After acceptance, bounding boxes from agreeing detectors are merged by
confidence-weighted average.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Dict

import numpy as np

from src.yolo_detector import Detection

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# IoU helper
# ──────────────────────────────────────────────────────────────────────────────

def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _merge_boxes(
    detections: List[Detection],
) -> List[float]:
    """
    Compute a confidence-weighted average bounding box from multiple detections.
    """
    if len(detections) == 1:
        return detections[0].bbox

    weights = np.array([d.confidence for d in detections])
    weights /= weights.sum() + 1e-9

    boxes = np.array([d.bbox for d in detections])   # (N, 4)
    merged = (weights[:, np.newaxis] * boxes).sum(axis=0)
    return merged.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Fusion engine
# ──────────────────────────────────────────────────────────────────────────────

class DetectionFusion:
    """
    Fuses YOLO + DETR + RCNN detections into a final accepted detection list.

    Parameters
    ----------
    min_model_agreement : int
        Minimum number of detectors that must agree on a detection.
    iou_merge_threshold : float
        IoU above which two detections are considered the same region.
    high_conf_threshold : float
        A single detector with confidence above this is accepted regardless of
        agreement (replaces GNN override logic).
    geometry_threshold : float
        Kept for backward compatibility; unused in acceptance rules.
    """

    def __init__(
        self,
        min_model_agreement: int = 2,
        iou_merge_threshold: float = 0.40,
        gnn_override_threshold: float = 0.85,   # repurposed as high_conf_threshold
        geometry_threshold: float = 0.60,
    ) -> None:
        self.min_agreement = min_model_agreement
        self.iou_threshold = iou_merge_threshold
        self.high_conf_threshold = gnn_override_threshold
        self.geometry_threshold = geometry_threshold

    # ── main entry ────────────────────────────────────────────────────────────

    def fuse(
        self,
        yolo_detections: List[Detection],
        detr_detections: List[Detection],
        rcnn_detections: List[Detection],
    ) -> List[Detection]:
        """
        Merge all three detector outputs into a final accepted detection list.

        Parameters
        ----------
        yolo_detections : output of YOLODetector.detect()
        detr_detections : output of DetrDetector.detect()
        rcnn_detections : output of RCNNDetector.detect()

        Returns
        -------
        List[Detection]  — deduplicated and merged.
        """
        all_raw = yolo_detections + detr_detections + rcnn_detections

        # ── Step 1: Group by label ───────────────────────────────────────────
        groups: dict[str, List[Detection]] = defaultdict(list)
        for det in all_raw:
            groups[det.label].append(det)

        # ── Step 2: Within each label, cluster overlapping boxes ─────────────
        candidates: List[List[Detection]] = []
        for label, dets in groups.items():
            clusters = self._cluster_boxes(dets)
            candidates.extend(clusters)

        # ── Step 3: Apply acceptance rules ───────────────────────────────────
        accepted: List[Detection] = []
        accepted_n_models: List[int] = []   # parallel list — model count per accepted det
        for cluster in candidates:
            label = cluster[0].label
            n_models = len({d.source for d in cluster})
            best_conf = max(d.confidence for d in cluster)

            rule_agreement  = n_models >= self.min_agreement
            # Strict consensus mode: when all 3 models are required,
            # do not allow single/high-confidence overrides.
            rule_high_conf  = (
                self.min_agreement < 3
                and best_conf >= self.high_conf_threshold
            )

            if not (rule_agreement or rule_high_conf):
                logger.debug(
                    f"Rejected '{label}': models={n_models}, best_conf={best_conf:.2f}"
                )
                continue

            merged_box = _merge_boxes(cluster)
            avg_conf = float(np.mean([d.confidence for d in cluster]))

            accepted.append(Detection(
                label=label,
                bbox=merged_box,
                confidence=avg_conf,
                source="fused",
                verified_by_gnn=False,
                gnn_score=0.0,
            ))
            accepted_n_models.append(n_models)

        # ── Step 5: Final cross-label NMS to remove spatial duplicates ─────────
        # Multi-model agreement detections are prioritised over single-model
        # high-confidence detections so that e.g. YOLO+DETR→ursa_major is never
        # suppressed by RCNN-alone→taurus even when RCNN has higher confidence.
        accepted, accepted_n_models = self._nms(
            accepted, accepted_n_models, self.iou_threshold
        )

        # Sort: more model agreement first, then by confidence within same count.
        paired_out = sorted(
            zip(accepted, accepted_n_models),
            key=lambda x: (x[1], x[0].confidence),
            reverse=True,
        )
        return [d for d, _ in paired_out]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _cluster_boxes(self, dets: List[Detection]) -> List[List[Detection]]:
        """
        Group detections with the same label whose boxes overlap ≥ iou_threshold
        into clusters using union-find (connected components).

        Unlike greedy single-link, this correctly handles transitive overlaps:
        if A↔B and B↔C overlap but A↔C do not, all three end up in one cluster.
        """
        if not dets:
            return []

        n = len(dets)
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            parent[_find(x)] = _find(y)

        for i in range(n):
            for j in range(i + 1, n):
                if _iou(dets[i].bbox, dets[j].bbox) >= self.iou_threshold:
                    _union(i, j)

        groups: Dict[int, List[Detection]] = defaultdict(list)
        for i in range(n):
            groups[_find(i)].append(dets[i])

        return list(groups.values())

    @staticmethod
    def _nms(
        dets: List[Detection],
        n_models: List[int],
        iou_thresh: float,
    ) -> tuple[List[Detection], List[int]]:
        """
        Greedy Non-Maximum Suppression to remove spatial duplicates across labels.

        Sorting priority: (n_models DESC, confidence DESC) — a detection backed
        by more models always beats a single-model high-confidence one.  This
        prevents e.g. RCNN-alone/taurus from suppressing YOLO+DETR/ursa_major
        simply because RCNN's confidence score is higher.

        Same-label detections are never suppressed — they represent distinct
        sky regions and were already deduplicated by the clustering step.
        """
        if not dets:
            return [], []

        # Zip, sort by (n_models DESC, confidence DESC), then unzip.
        paired = sorted(
            zip(dets, n_models),
            key=lambda x: (x[1], x[0].confidence),
            reverse=True,
        )
        dets_sorted = [p[0] for p in paired]
        nm_sorted   = [p[1] for p in paired]

        keep_dets:  List[Detection] = []
        keep_nm:    List[int]       = []
        suppressed = [False] * len(dets_sorted)

        for i, d in enumerate(dets_sorted):
            if suppressed[i]:
                continue
            keep_dets.append(d)
            keep_nm.append(nm_sorted[i])
            for j, other in enumerate(dets_sorted[i + 1:], start=i + 1):
                if suppressed[j]:
                    continue
                # Only suppress across different labels; same-label detections
                # in distinct regions should both be kept.
                if d.label == other.label:
                    continue
                if _iou(d.bbox, other.bbox) >= iou_thresh:
                    suppressed[j] = True

        return keep_dets, keep_nm


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def fuse_detections(
    yolo: List[Detection],
    detr: List[Detection],
    rcnn: List[Detection],
    cfg: dict | None = None,
) -> List[Detection]:
    cfg = cfg or {}
    engine = DetectionFusion(
        min_model_agreement=cfg.get("min_model_agreement", 2),
        iou_merge_threshold=cfg.get("iou_merge_threshold", 0.40),
        gnn_override_threshold=cfg.get("gnn_override_threshold", 0.85),
        geometry_threshold=cfg.get("geometry_threshold", 0.60),
    )
    return engine.fuse(yolo, detr, rcnn)
