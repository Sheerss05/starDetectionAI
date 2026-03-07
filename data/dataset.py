"""
dataset.py
───────────
Dataset utilities for constellation detection.

Provides helpers to:
  • Convert between annotation formats (COCO ↔ YOLO)
  • Build graph dataset (.pt files) from annotated images + star extractions
  • Validate dataset integrity
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def coco_to_yolo(
    coco_json_path: str,
    images_dir: str,
    output_labels_dir: str,
) -> None:
    """Convert COCO annotation JSON → YOLO .txt label files.

    If the destination is on OneDrive and cloud-only (files-on-demand), the
    function automatically falls back to writing inside a local folder at
    ``C:\\Users\\<user>\\AppData\\Local\\starAIv2_labels\\`` and prints
    instructions for copying the output to the intended destination.
    """
    import os
    import subprocess
    import sys
    from collections import defaultdict

    def _try_write(path: Path, text: str) -> bool:
        """Return True if writing *text* to *path* succeeds and the file is
        visible on disk afterwards."""
        try:
            path.write_text(text, encoding="utf-8")
            return path.exists() and path.stat().st_size > 0
        except OSError:
            return False

    out_dir = Path(output_labels_dir).resolve()

    # Pin the directory locally on Windows (OneDrive "Always keep on this device")
    if sys.platform == "win32":
        subprocess.run(["attrib", "+P", str(out_dir), "/S", "/D"], capture_output=True)

    os.makedirs(str(out_dir), exist_ok=True)

    # Probe whether Python can actually write to the intended directory
    _probe = out_dir / "__probe__.txt"
    _onedrive_fallback = not _try_write(_probe, "probe")
    if _probe.exists():
        _probe.unlink(missing_ok=True)

    if _onedrive_fallback:
        # Fall back to a writable local path outside OneDrive
        import re
        split_name = re.sub(r"[^a-zA-Z0-9_-]", "_", out_dir.name)
        out_dir = Path(os.environ.get("LOCALAPPDATA", "C:/Users/Public")) / "starAIv2_labels" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[WARNING] OneDrive cloud-only directory detected.\n"
            f"          Labels are being written to the local fallback:\n"
            f"          {out_dir}\n"
            f"          After conversion, copy those .txt files into:\n"
            f"          {Path(output_labels_dir).resolve()}\n"
            f"          (or right-click that folder in File Explorer → "
            f"'Always keep on this device', then re-run)\n"
        )

    with open(coco_json_path) as f:
        coco = json.load(f)

    img_map: Dict[int, dict] = {img["id"]: img for img in coco["images"]}
    cat_map: Dict[int, int] = {cat["id"]: i for i, cat in enumerate(coco["categories"])}

    # Collect all lines per label file first (handles multiple boxes per image)
    label_lines: Dict[str, list] = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_info = img_map[ann["image_id"]]
        W, H = img_info["width"], img_info["height"]
        x, y, w, h = ann["bbox"]
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        nw, nh = w / W, h / H
        cls = cat_map[ann["category_id"]]
        stem = Path(img_info["file_name"]).stem
        label_lines[stem].append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Write each label file once
    written = 0
    for stem, lines in label_lines.items():
        label_path = out_dir / f"{stem}.txt"
        label_path.write_text("".join(lines), encoding="utf-8")
        written += 1

    logger.info(f"COCO→YOLO conversion done. {written} label files written to {out_dir}")
    print(f"[convert] {written} label files → {out_dir}")


def yolo_to_coco(
    images_dir: str,
    labels_dir: str,
    class_names: List[str],
    output_json: str,
) -> None:
    """Convert YOLO label .txt files → COCO-format annotation JSON."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    coco_images, coco_annotations = [], []
    ann_id = 1

    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        img_id = len(coco_images) + 1
        coco_images.append({
            "id": img_id, "file_name": img_path.name,
            "width": W, "height": H,
        })
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        with open(label_path) as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, nw, nh = int(parts[0]), *map(float, parts[1:])
                x = (cx - nw / 2) * W
                y = (cy - nh / 2) * H
                w, h = nw * W, nh * H
                coco_annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cls,
                    "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                })
                ann_id += 1

    coco_out = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    with open(output_json, "w") as f:
        json.dump(coco_out, f, indent=2)
    logger.info(f"YOLO→COCO conversion done.  JSON written to {output_json}")


def build_graph_dataset(
    images_dir: str,
    coco_json_path: str,
    class_names: List[str],
    output_dir: str,
    star_cfg: Optional[dict] = None,
    graph_cfg: Optional[dict] = None,
    image_size: Tuple[int, int] = (640, 640),
) -> None:
    """
    For each annotated constellation bounding box, extract a star sub-graph
    and save it as a PyG Data .pt file.

    Layout:
      output_dir/graphs/<label>/<img_stem>_<ann_id>.pt
    """
    from src.preprocessing import Preprocessor
    from src.star_extraction import StarExtractor
    from src.graph_construction import StarGraphBuilder

    preprocessor = Preprocessor(target_size=image_size)
    extractor = StarExtractor(**(star_cfg or {}))
    graph_builder = StarGraphBuilder(**(graph_cfg or {}), image_size=image_size)

    images_dir = Path(images_dir)
    out_dir = Path(output_dir)

    with open(coco_json_path) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    cat_map = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    saved = 0
    for ann in coco.get("annotations", []):
        img_info = img_map[ann["image_id"]]
        img_path = images_dir / img_info["file_name"]
        label_str = cat_map.get(ann["category_id"], str(ann["category_id"]))

        if label_str not in class_names:
            continue

        processed, _ = preprocessor.process(img_path)
        stars = extractor.extract(processed)

        x, y, w, h = ann["bbox"]
        orig_W, orig_H = img_info["width"], img_info["height"]
        tH, tW = image_size
        sx, sy = tW / orig_W, tH / orig_H
        bbox = [x * sx, y * sy, (x + w) * sx, (y + h) * sy]

        subgraph = graph_builder.build_subgraph(stars, bbox)
        if subgraph is None:
            continue

        cls_idx = class_names.index(label_str)
        subgraph.y = torch.tensor([cls_idx], dtype=torch.long)

        save_dir = out_dir / "graphs" / label_str
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(subgraph, save_dir / f"{img_path.stem}_{ann['id']}.pt")
        saved += 1

    logger.info(f"Graph dataset built: {saved} graphs saved to {out_dir}")


def build_split_json(graph_root: str, val_ratio: float = 0.2) -> None:
    """Write train/val split.json for the graph dataset."""
    import random

    graph_root = Path(graph_root)
    all_files: Dict[str, List[str]] = {}

    for label_dir in sorted((graph_root / "graphs").iterdir()):
        if label_dir.is_dir():
            all_files[label_dir.name] = [str(f) for f in sorted(label_dir.glob("*.pt"))]

    train_files, val_files = [], []
    for label, files in all_files.items():
        random.shuffle(files)
        cut = max(1, int(len(files) * (1 - val_ratio)))
        train_files.extend(files[:cut])
        val_files.extend(files[cut:])

    split_path = graph_root / "split.json"
    with open(split_path, "w") as f:
        json.dump({"train": train_files, "val": val_files}, f, indent=2)
    logger.info(
        f"Split written to {split_path}: "
        f"{len(train_files)} train, {len(val_files)} val"
    )
