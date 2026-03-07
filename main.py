"""
main.py
────────
Command-line entry point for the Constellation Recognition AI.

Usage
─────
  # Run inference on a single image
  python main.py infer --image path/to/sky.jpg

  # Run inference and save annotated result
  python main.py infer --image sky.jpg --save output/result.jpg

  # Run inference in CPU mode
  python main.py infer --image sky.jpg --device cpu

  # Train YOLO
  python main.py train --model yolo --config configs/config.yaml

  # Train DETR
  python main.py train --model detr --config configs/config.yaml

  # Train GNN
  python main.py train --model gnn --config configs/config.yaml

  # Build graph dataset from annotated images
  python main.py build-graphs --coco data/annotations.json --images data/images/

  # Convert COCO labels to YOLO format
  python main.py convert --direction coco2yolo --coco data/ann.json --out data/labels/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ──────────────────────────────────────────────────────────────────────────────

def cmd_infer(args) -> None:
    """Run the full detection pipeline on one image."""
    from src.pipeline import ConstellationPipeline

    pipeline = ConstellationPipeline(
        config_path=args.config,
        device=args.device,
    )

    result = pipeline.run(args.image)
    print(result.summarise())

    if args.save:
        pipeline.visualise(result, save_path=args.save)
        print(f"\nAnnotated image saved to: {args.save}")

    if args.json:
        out = json.dumps(result.to_json(), indent=2)
        if args.json == "-":
            print(out)
        else:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(out)
            print(f"JSON results saved to: {args.json}")


def cmd_train(args) -> None:
    """Launch training for a specified model."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model == "yolo":
        from training.train_yolo import train
    elif args.model == "detr":
        from training.train_detr import train
    elif args.model == "gnn":
        from training.train_gnn import train
    else:
        logger.error(f"Unknown model: {args.model}.  Choose: yolo | detr | gnn")
        sys.exit(1)

    train(cfg)


def cmd_build_graphs(args) -> None:
    """Build graph dataset (.pt files) from annotated images."""
    from data.dataset import build_graph_dataset, build_split_json
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    class_names = cfg.get("constellations", [])
    output_dir  = args.output or "data/graph_dataset"

    build_graph_dataset(
        images_dir=args.images,
        coco_json_path=args.coco,
        class_names=class_names,
        output_dir=output_dir,
        image_size=tuple(cfg.get("preprocessing", {}).get("target_size", [640, 640])),
    )

    build_split_json(output_dir)


def cmd_convert(args) -> None:
    """Convert between annotation formats."""
    from data.dataset import coco_to_yolo, yolo_to_coco
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.direction == "coco2yolo":
        if not args.coco or not args.out:
            logger.error("--coco and --out required for coco2yolo")
            sys.exit(1)
        coco_to_yolo(args.coco, args.images or "", args.out)
    elif args.direction == "yolo2coco":
        if not args.images or not args.labels or not args.out:
            logger.error("--images, --labels, and --out required for yolo2coco")
            sys.exit(1)
        yolo_to_coco(
            args.images, args.labels,
            class_names=cfg.get("constellations", []),
            output_json=args.out,
        )
    else:
        logger.error(f"Unknown direction: {args.direction}.  Use coco2yolo or yolo2coco")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="starAI",
        description="Constellation Recognition AI — YOLO + DETR + GNN hybrid pipeline",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to configs/config.yaml",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── infer ──────────────────────────────────────────────────────────────────
    infer = sub.add_parser("infer", help="Run constellation detection on an image")
    infer.add_argument("--image", required=True, help="Path to input sky image")
    infer.add_argument("--save",  default=None,  help="Save annotated output image")
    infer.add_argument("--json",  default=None,  help="Save JSON results (- for stdout)")
    infer.add_argument("--device", default=None, help="Device override: cuda | cpu")

    # ── train ──────────────────────────────────────────────────────────────────
    trn = sub.add_parser("train", help="Train one of the models")
    trn.add_argument("--model", required=True, choices=["yolo", "detr", "gnn"])

    # ── build-graphs ───────────────────────────────────────────────────────────
    bg = sub.add_parser("build-graphs", help="Build GNN graph dataset from annotations")
    bg.add_argument("--coco",   required=True, help="COCO annotation JSON path")
    bg.add_argument("--images", required=True, help="Images directory")
    bg.add_argument("--output", default=None,  help="Output directory (default: data/graph_dataset)")

    # ── convert ────────────────────────────────────────────────────────────────
    cv = sub.add_parser("convert", help="Convert annotation formats")
    cv.add_argument("--direction", required=True, choices=["coco2yolo", "yolo2coco"])
    cv.add_argument("--coco",   default=None, help="COCO JSON input")
    cv.add_argument("--images", default=None, help="Images directory")
    cv.add_argument("--labels", default=None, help="YOLO labels directory (yolo2coco)")
    cv.add_argument("--out",    default=None, help="Output path")

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "infer":        cmd_infer,
        "train":        cmd_train,
        "build-graphs": cmd_build_graphs,
        "convert":      cmd_convert,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
