"""
Single-script evaluation for YOLO, Faster R-CNN, and DETR object detection models.

This script:
1) Loads existing trained weights (.pt) for all three models.
2) Runs inference on ONE shared test dataset (images + YOLO-format labels).
3) Calculates real metrics:
   - mAP@0.5
   - Precision
   - Recall
   - F1-Score
   - AP-Small (COCO-style small objects: area < 32^2 pixels)
   - Average Confidence Score of correct detections (TP only)
   - Latency (ms per image)
   - Number of Parameters (Millions)
4) Saves a comparison table to:
   - comparison_results.csv
   - comparison_results.xls

Important:
- No retraining is performed.
- GPU is used automatically if available; otherwise CPU.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ultralytics import YOLO
from src.detr_detector import DetrDetector
from src.rcnn_detector import RCNNDetector


SMALL_OBJECT_AREA = 32 * 32  # COCO small-object definition in pixel^2


@dataclass
class Pred:
    image_id: str
    cls: int
    conf: float
    bbox: np.ndarray  # [x1, y1, x2, y2]


@dataclass
class GT:
    image_id: str
    cls: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    area: float


def get_device() -> str:
    """Use GPU if available, else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def xywhn_to_xyxy(label_row: np.ndarray, img_w: int, img_h: int) -> Tuple[int, np.ndarray]:
    """Convert one YOLO label row [cls, cx, cy, w, h] (normalized) to xyxy pixels."""
    cls = int(label_row[0])
    cx, cy, w, h = label_row[1:].tolist()
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return cls, np.array([x1, y1, x2, y2], dtype=np.float32)


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two xyxy boxes."""
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """
    Compute AP using interpolated precision-recall envelope.

    This is the standard area-under-PR-curve style AP used in many detection
    evaluations for a fixed IoU threshold.
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def load_dataset(image_dir: Path, label_dir: Path) -> Tuple[List[Path], Dict[str, List[GT]], int]:
    """Load test image paths + YOLO-format labels as per-image GT boxes."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in exts])
    if not image_paths:
        raise FileNotFoundError(f"No test images found in: {image_dir}")

    gt_by_image: Dict[str, List[GT]] = {}
    max_cls = -1

    for img_path in image_paths:
        image_id = img_path.stem
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        label_path = label_dir / f"{image_id}.txt"
        gts: List[GT] = []

        if label_path.exists() and label_path.stat().st_size > 0:
            rows = np.loadtxt(label_path, dtype=np.float32)
            rows = np.atleast_2d(rows)
            for row in rows:
                cls, box = xywhn_to_xyxy(row, img_w, img_h)
                area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
                gts.append(GT(image_id=image_id, cls=cls, bbox=box, area=area))
                max_cls = max(max_cls, cls)

        gt_by_image[image_id] = gts

    num_classes = max_cls + 1 if max_cls >= 0 else 1
    return image_paths, gt_by_image, num_classes


def load_class_names_from_coco(annotation_path: Optional[Path]) -> List[str]:
    """Load ordered class names from COCO categories sorted by id."""
    if annotation_path is None or not annotation_path.exists():
        return []

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))
    if not categories:
        return []

    # Normalize into a dense id space [0..N-1]. If IDs are sparse, fallback to names order.
    ids = [int(c["id"]) for c in categories]
    dense = ids == list(range(min(ids), min(ids) + len(ids))) and min(ids) in (0, 1)

    if dense and min(ids) == 1:
        # Shift 1-based ids into 0-based list positions.
        names = [""] * len(categories)
        for c in categories:
            names[int(c["id"]) - 1] = str(c["name"]).strip()
        return names

    if dense and min(ids) == 0:
        names = [""] * len(categories)
        for c in categories:
            names[int(c["id"])] = str(c["name"]).strip()
        return names

    return [str(c["name"]).strip() for c in categories]


def build_name_to_id(class_names: List[str]) -> Dict[str, int]:
    """Create case-insensitive class-name lookup."""
    mapping: Dict[str, int] = {}
    for idx, name in enumerate(class_names):
        key = str(name).strip().lower()
        if key:
            mapping[key] = idx
    return mapping


def remap_pred_class(
    raw_label: str,
    fallback_id: Optional[int],
    class_name_to_id: Dict[str, int],
    num_classes: int,
) -> Optional[int]:
    """Map a detector output label into evaluator class IDs."""
    key = str(raw_label).strip().lower()
    if key in class_name_to_id:
        return class_name_to_id[key]

    if key.isdigit():
        cid = int(key)
        if 0 <= cid < num_classes:
            return cid

    if fallback_id is not None and 0 <= fallback_id < num_classes:
        return int(fallback_id)

    return None


def evaluate_predictions(
    preds: List[Pred],
    gt_by_image: Dict[str, List[GT]],
    num_classes: int,
    iou_thr: float = 0.5,
    small_only: bool = False,
) -> Tuple[float, int, int, int, List[float]]:
    """
    Evaluate predictions and return:
    - mAP@IoU
    - TP, FP, FN (global counts)
    - confidence values of true positives

    If small_only=True, GT pool is restricted to small objects (area < 32^2).
    """
    aps: List[float] = []
    global_tp = 0
    global_fp = 0
    tp_confidences: List[float] = []

    # Build GT lookup per class/image and matched flags
    gt_pool: Dict[int, Dict[str, List[GT]]] = {c: {} for c in range(num_classes)}
    total_gt_per_class = {c: 0 for c in range(num_classes)}

    for image_id, gts in gt_by_image.items():
        for gt in gts:
            if small_only and gt.area >= SMALL_OBJECT_AREA:
                continue
            gt_pool.setdefault(gt.cls, {}).setdefault(image_id, []).append(gt)
            total_gt_per_class[gt.cls] = total_gt_per_class.get(gt.cls, 0) + 1

    # Per-class AP computation from ranked predictions
    for cls in range(num_classes):
        cls_preds = [p for p in preds if p.cls == cls]
        cls_preds.sort(key=lambda x: x.conf, reverse=True)

        n_gt = total_gt_per_class.get(cls, 0)
        if n_gt == 0:
            continue

        matched = {
            img_id: np.zeros(len(gt_pool.get(cls, {}).get(img_id, [])), dtype=bool)
            for img_id in gt_pool.get(cls, {})
        }

        tp = np.zeros(len(cls_preds), dtype=np.float32)
        fp = np.zeros(len(cls_preds), dtype=np.float32)

        for i, pred in enumerate(cls_preds):
            gt_list = gt_pool.get(cls, {}).get(pred.image_id, [])
            if not gt_list:
                fp[i] = 1.0
                continue

            ious = np.array([bbox_iou_xyxy(pred.bbox, gt.bbox) for gt in gt_list], dtype=np.float32)
            best_idx = int(np.argmax(ious)) if ious.size else -1
            best_iou = float(ious[best_idx]) if ious.size else 0.0

            if best_iou >= iou_thr and not matched[pred.image_id][best_idx]:
                tp[i] = 1.0
                matched[pred.image_id][best_idx] = True
                global_tp += 1
                tp_confidences.append(pred.conf)
            else:
                fp[i] = 1.0
                global_fp += 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        rec = cum_tp / max(n_gt, 1)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

        aps.append(voc_ap(rec, prec))

    total_gt = sum(total_gt_per_class.values())
    global_fn = max(total_gt - global_tp, 0)

    map50 = float(np.mean(aps)) if aps else 0.0
    return map50, global_tp, global_fp, global_fn, tp_confidences


def count_parameters_m(model: torch.nn.Module) -> float:
    """Count total model parameters in millions."""
    return float(sum(p.numel() for p in model.parameters()) / 1e6)


def class_agnostic_iou_stats(
    preds: List[Pred],
    gt_by_image: Dict[str, List[GT]],
    iou_thr: float = 0.5,
) -> Dict[str, float]:
    """Compute IoU matching stats while ignoring class labels."""
    preds_by_image: Dict[str, List[Pred]] = {}
    for p in preds:
        preds_by_image.setdefault(p.image_id, []).append(p)

    tp = 0
    fp = 0
    total_gt = 0

    for image_id, gts in gt_by_image.items():
        total_gt += len(gts)
        image_preds = sorted(preds_by_image.get(image_id, []), key=lambda x: x.conf, reverse=True)
        matched = np.zeros(len(gts), dtype=bool)

        for pred in image_preds:
            if not gts:
                fp += 1
                continue

            ious = np.array([bbox_iou_xyxy(pred.bbox, gt.bbox) for gt in gts], dtype=np.float32)
            best_idx = int(np.argmax(ious)) if ious.size else -1
            best_iou = float(ious[best_idx]) if ious.size else 0.0

            if best_iou >= iou_thr and not matched[best_idx]:
                matched[best_idx] = True
                tp += 1
            else:
                fp += 1

    fn = max(total_gt - tp, 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "agnostic_tp": float(tp),
        "agnostic_fp": float(fp),
        "agnostic_fn": float(fn),
        "agnostic_precision": float(precision),
        "agnostic_recall": float(recall),
    }


def run_model_on_dataset(
    image_paths: List[Path],
    gt_by_image: Dict[str, List[GT]],
    num_classes: int,
    infer_fn: Callable[[np.ndarray], List[Pred]],
    diag_name: Optional[str] = None,
) -> Dict[str, float]:
    """Generic evaluator for one detector given an inference callback."""
    preds: List[Pred] = []
    latencies_ms: List[float] = []

    for img_path in image_paths:
        image_id = img_path.stem
        image_np = np.array(Image.open(img_path).convert("RGB"))

        t0 = time.time()
        out = infer_fn(image_np)
        t1 = time.time()
        latencies_ms.append((t1 - t0) * 1000.0)

        for p in out:
            p.image_id = image_id
            preds.append(p)

    # Main metrics at IoU=0.5
    map50, tp, fp, fn, tp_confs = evaluate_predictions(
        preds=preds,
        gt_by_image=gt_by_image,
        num_classes=num_classes,
        iou_thr=0.5,
        small_only=False,
    )

    # AP-Small at IoU=0.5 restricted to small GT objects
    ap_small, _, _, _, _ = evaluate_predictions(
        preds=preds,
        gt_by_image=gt_by_image,
        num_classes=num_classes,
        iou_thr=0.5,
        small_only=True,
    )

    # Precision and Recall from TP/FP/FN
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 from Precision and Recall
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Mean confidence of correct detections (TP only)
    avg_conf_tp = float(np.mean(tp_confs)) if tp_confs else 0.0

    latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    if diag_name:
        agnostic = class_agnostic_iou_stats(preds=preds, gt_by_image=gt_by_image, iou_thr=0.5)
        print(
            f"{diag_name} class-agnostic IoU@0.5: "
            f"P={agnostic['agnostic_precision']:.4f}, R={agnostic['agnostic_recall']:.4f}, "
            f"TP={int(agnostic['agnostic_tp'])}, FP={int(agnostic['agnostic_fp'])}, FN={int(agnostic['agnostic_fn'])}"
        )

    return {
        "mAP@0.5": map50,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AP-Small": ap_small,
        "Avg Confidence (TP)": avg_conf_tp,
        "Latency (ms/img)": latency_ms,
    }


def evaluate_yolo(
    weights_path: Path,
    image_paths: List[Path],
    gt_by_image: Dict[str, List[GT]],
    num_classes: int,
    class_names: List[str],
    class_name_to_id: Dict[str, int],
    device: str,
) -> Dict[str, float]:
    """Evaluate YOLO via Ultralytics YOLO API."""
    model = YOLO(str(weights_path))
    model.to(device)

    dropped_unmapped = 0
    total_raw = 0

    def infer(image_np: np.ndarray) -> List[Pred]:
        nonlocal dropped_unmapped, total_raw
        res = model.predict(source=image_np, device=device, verbose=False)
        out: List[Pred] = []
        names_obj = getattr(model, "names", {})

        if isinstance(names_obj, dict):
            yolo_name_map = {int(k): str(v) for k, v in names_obj.items()}
        else:
            yolo_name_map = {i: str(v) for i, v in enumerate(list(names_obj))}

        for r in res:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            conf = r.boxes.conf.detach().cpu().numpy()
            cls = r.boxes.cls.detach().cpu().numpy().astype(int)
            for b, c, cl in zip(xyxy, conf, cls):
                total_raw += 1
                raw_name = yolo_name_map.get(int(cl), str(cl))
                mapped = remap_pred_class(raw_name, int(cl), class_name_to_id, num_classes)
                if mapped is None:
                    dropped_unmapped += 1
                    continue
                out.append(Pred(image_id="", cls=int(mapped), conf=float(c), bbox=b.astype(np.float32)))
        return out

    metrics = run_model_on_dataset(image_paths, gt_by_image, num_classes, infer)
    metrics["Params (M)"] = count_parameters_m(model.model)
    kept = total_raw - dropped_unmapped
    print(f"YOLO label remap: raw={total_raw}, kept={kept}, dropped_unmapped={dropped_unmapped}")
    return metrics


def evaluate_fasterrcnn(
    weights_path: Path,
    image_paths: List[Path],
    gt_by_image: Dict[str, List[GT]],
    num_classes: int,
    class_names: List[str],
    class_name_to_id: Dict[str, int],
    device: str,
) -> Dict[str, float]:
    """Evaluate Faster R-CNN (PyTorch/torchvision-based wrapper)."""
    detector = RCNNDetector(
        model_weights=str(weights_path),
        device=device,
        num_classes=num_classes,
        class_names=class_names,
    )

    dropped_unmapped = 0
    total_raw = 0

    def infer(image_np: np.ndarray) -> List[Pred]:
        nonlocal dropped_unmapped, total_raw
        dets = detector.detect(image_np)
        out: List[Pred] = []
        for d in dets:
            total_raw += 1
            mapped = remap_pred_class(d.label, None, class_name_to_id, num_classes)
            if mapped is None:
                try:
                    mapped = remap_pred_class(d.label, int(d.label), class_name_to_id, num_classes)
                except (TypeError, ValueError):
                    mapped = None
            if mapped is None:
                dropped_unmapped += 1
                continue
            out.append(
                Pred(
                    image_id="",
                    cls=int(mapped),
                    conf=float(d.confidence),
                    bbox=np.array(d.bbox, dtype=np.float32),
                )
            )
        return out

    metrics = run_model_on_dataset(
        image_paths,
        gt_by_image,
        num_classes,
        infer,
        diag_name="Faster R-CNN",
    )
    metrics["Params (M)"] = count_parameters_m(detector.model)
    kept = total_raw - dropped_unmapped
    print(f"Faster R-CNN label remap: raw={total_raw}, kept={kept}, dropped_unmapped={dropped_unmapped}")
    return metrics


def evaluate_detr(
    weights_path: Path,
    image_paths: List[Path],
    gt_by_image: Dict[str, List[GT]],
    num_classes: int,
    class_names: List[str],
    class_name_to_id: Dict[str, int],
    device: str,
) -> Dict[str, float]:
    """Evaluate DETR (PyTorch/Transformers-based wrapper)."""
    detector = DetrDetector(
        model_weights=str(weights_path),
        device=device,
        num_classes=num_classes,
        class_names=class_names,
    )

    dropped_unmapped = 0
    total_raw = 0

    def infer(image_np: np.ndarray) -> List[Pred]:
        nonlocal dropped_unmapped, total_raw
        dets = detector.detect(image_np)
        out: List[Pred] = []
        for d in dets:
            total_raw += 1
            mapped = remap_pred_class(d.label, None, class_name_to_id, num_classes)
            if mapped is None:
                try:
                    mapped = remap_pred_class(d.label, int(d.label), class_name_to_id, num_classes)
                except (TypeError, ValueError):
                    mapped = None
            if mapped is None:
                dropped_unmapped += 1
                continue
            out.append(
                Pred(
                    image_id="",
                    cls=int(mapped),
                    conf=float(d.confidence),
                    bbox=np.array(d.bbox, dtype=np.float32),
                )
            )
        return out

    metrics = run_model_on_dataset(image_paths, gt_by_image, num_classes, infer)
    metrics["Params (M)"] = count_parameters_m(detector.model)
    kept = total_raw - dropped_unmapped
    print(f"DETR label remap: raw={total_raw}, kept={kept}, dropped_unmapped={dropped_unmapped}")
    return metrics


def save_results(df: pd.DataFrame) -> None:
    """Save results to CSV and XLS."""
    df.to_csv("comparison_results.csv", index=False)

    # Primary .xls export path
    try:
        df.to_excel("comparison_results.xls", index=False, engine="xlwt")
    except Exception:
        # Fallback: TSV content with .xls extension for Excel compatibility if xlwt is unavailable.
        df.to_csv("comparison_results.xls", index=False, sep="\t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO, Faster R-CNN, and DETR on one shared test set.")

    parser.add_argument("--test-images", type=Path, default=Path("./dataset/test/images"))
    parser.add_argument("--test-labels", type=Path, default=Path("./dataset/test/labels"))
    parser.add_argument("--test-annotations", type=Path, default=Path("data/constellation_dataset/annotations/test.json"))

    parser.add_argument("--yolo-weights", type=Path, default=Path("models/yolo/constellation_yolo.pt"))
    parser.add_argument("--rcnn-weights", type=Path, default=Path("models/rcnn/constellation_rcnn.pt"))
    parser.add_argument("--detr-weights", type=Path, default=Path("models/detr/detr_checkpoint.pt"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Optional fallback to repo's existing structure if user default path is absent.
    test_images = args.test_images
    test_labels = args.test_labels
    if not test_images.exists() or not test_labels.exists():
        alt_images = Path("data/constellation_dataset/images/test")
        alt_labels = Path("data/constellation_dataset/labels/test")
        if alt_images.exists() and alt_labels.exists():
            test_images = alt_images
            test_labels = alt_labels

    test_annotations = args.test_annotations
    if not test_annotations.exists():
        alt_ann = Path("data/constellation_dataset/annotations/test.json")
        if alt_ann.exists():
            test_annotations = alt_ann

    if not test_images.exists() or not test_labels.exists():
        raise FileNotFoundError(
            f"Test dataset paths not found. Checked: {args.test_images}, {args.test_labels}"
        )

    if not args.yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_weights}")
    if not args.rcnn_weights.exists():
        raise FileNotFoundError(f"Faster R-CNN weights not found: {args.rcnn_weights}")
    if not args.detr_weights.exists():
        raise FileNotFoundError(f"DETR weights not found: {args.detr_weights}")

    device = get_device()
    print(f"Running evaluation on device: {device}")

    image_paths, gt_by_image, label_num_classes = load_dataset(test_images, test_labels)

    class_names = load_class_names_from_coco(test_annotations)
    if class_names and len(class_names) >= label_num_classes:
        num_classes = len(class_names)
        print(
            f"Loaded {len(image_paths)} test images | classes from COCO categories: {num_classes} "
            f"(labels observed: {label_num_classes})"
        )
    else:
        num_classes = label_num_classes
        class_names = [str(i) for i in range(num_classes)]
        print(
            f"Loaded {len(image_paths)} test images | classes detected from labels: {num_classes} "
            "(COCO categories unavailable or mismatched)"
        )

    class_name_to_id = build_name_to_id(class_names)
    print(f"Class mapping source: {'COCO test annotations' if test_annotations.exists() else 'YOLO labels only'}")

    print("Evaluating YOLO...")
    yolo_metrics = evaluate_yolo(
        args.yolo_weights,
        image_paths,
        gt_by_image,
        num_classes,
        class_names,
        class_name_to_id,
        device,
    )

    print("Evaluating Faster R-CNN...")
    rcnn_metrics = evaluate_fasterrcnn(
        args.rcnn_weights,
        image_paths,
        gt_by_image,
        num_classes,
        class_names,
        class_name_to_id,
        device,
    )

    print("Evaluating DETR...")
    detr_metrics = evaluate_detr(
        args.detr_weights,
        image_paths,
        gt_by_image,
        num_classes,
        class_names,
        class_name_to_id,
        device,
    )

    rows = [
        {"Model": "YOLO", **yolo_metrics},
        {"Model": "Faster R-CNN", **rcnn_metrics},
        {"Model": "DETR", **detr_metrics},
    ]
    df = pd.DataFrame(rows)

    # Keep output order aligned with requirement list.
    ordered_cols = [
        "Model",
        "mAP@0.5",
        "Precision",
        "Recall",
        "F1-Score",
        "AP-Small",
        "Avg Confidence (TP)",
        "Latency (ms/img)",
        "Params (M)",
    ]
    df = df[ordered_cols]

    save_results(df)

    print("\nFinal Comparison Table")
    print(df.to_string(index=False))
    print("\nSaved: comparison_results.csv")
    print("Saved: comparison_results.xls")


if __name__ == "__main__":
    main()
