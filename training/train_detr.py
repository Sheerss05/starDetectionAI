"""
train_detr.py
──────────────
Fine-tune a DETR (Detection Transformer) model on the constellation dataset.

The HuggingFace ``facebook/detr-resnet-50`` backbone is repurposed by replacing
its classification head with a num_classes-output head.

Dataset format expected
───────────────────────
  data/constellation_dataset/
    images/  {train,val}/*.jpg
    annotations/
      train.json    (COCO-style annotations)
      val.json

Run
───
  python training/train_detr.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_emergency_state = None


# ──────────────────────────────────────────────────────────────────────────────
# COCO-format dataset  (minimal implementation)
# ──────────────────────────────────────────────────────────────────────────────

class ConstellationDetrDataset(torch.utils.data.Dataset):
    """
    Minimal COCO-style dataset for DETR fine-tuning.

    Each item returns:
      pixel_values  : float tensor (3, H, W)
      labels        : dict with keys 'class_labels', 'boxes' (cx cy w h normalised)
    """

    def __init__(self, image_dir: str, annotation_file: str, processor) -> None:
        from PIL import Image as PILImage
        self.pil = PILImage
        self.image_dir = Path(image_dir)
        self.processor = processor

        with open(annotation_file) as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.annotations: dict[int, list] = {}
        for ann in coco.get("annotations", []):
            self.annotations.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img = self.pil.open(self.image_dir / img_info["file_name"]).convert("RGB")

        annots = self.annotations.get(img_id, [])
        W, H = img.size

        boxes, class_ids = [], []
        for ann in annots:
            x, y, w, h = ann["bbox"]
            # Convert COCO (x,y,w,h) → DETR (cx,cy,w,h) normalised
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw, nh = w / W, h / H
            boxes.append([cx, cy, nw, nh])
            class_ids.append(ann["category_id"])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "class_labels": torch.tensor(class_ids, dtype=torch.long),
            "image_id": torch.tensor([img_id]),
        }

        encoding = self.processor(images=img, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        return pixel_values, target


def _collate(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    global _emergency_state
    _emergency_state = None
    try:
        from transformers import DetrForObjectDetection, DetrImageProcessor
    except ImportError:
        raise ImportError("transformers not installed.  Run: pip install transformers")

    detr_cfg  = cfg.get("detr", {})
    train_cfg = cfg.get("train_detr", {})

    device = detr_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA unavailable — falling back to CPU.")

    num_classes  = int(detr_cfg.get("num_classes", 88))
    base_model   = detr_cfg.get("pretrained_base", "facebook/detr-resnet-50")
    data_root    = Path(train_cfg.get("data_root", "data/constellation_dataset"))
    epochs       = int(train_cfg.get("epochs", 150))
    batch_size   = int(train_cfg.get("batch_size", 8))
    lr           = float(train_cfg.get("lr", 1e-4))
    lr_backbone  = float(train_cfg.get("lr_backbone", 1e-5))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    num_workers  = int(train_cfg.get("num_workers", 0))

    # ── Build model ───────────────────────────────────────────────────────────
    processor = DetrImageProcessor.from_pretrained(base_model)
    model = DetrForObjectDetection.from_pretrained(
        base_model,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ann = data_root / "annotations" / "train.json"
    val_ann   = data_root / "annotations" / "val.json"

    if not train_ann.exists():
        logger.warning(f"Annotation file not found at {train_ann}.  Skipping training.")
        return

    train_ds = ConstellationDetrDataset(data_root / "images" / "train", train_ann, processor)
    val_ds   = ConstellationDetrDataset(data_root / "images" / "val", val_ann, processor) \
               if val_ann.exists() else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=_collate)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=_collate) if val_ds else None

    # ── Optimizer with backbone LR separation ─────────────────────────────────
    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": lr_backbone},
    ]
    optimizer = AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=max(1, epochs // 5), gamma=0.5)

    best_val_loss = float("inf")
    start_epoch   = 1
    out_weights   = Path(detr_cfg.get("model_weights", "models/detr/constellation_detr.pt"))
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_weights.parent / "detr_checkpoint.pt"

    # ── Resume from checkpoint if available ───────────────────────────────────
    if checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") and "param_groups" in ckpt["optimizer"]:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") and "last_epoch" in ckpt["scheduler"]:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed at epoch {start_epoch}, best val loss so far: {best_val_loss:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        logger.info(f"Epoch {epoch} — train loss: {avg_loss:.4f}")

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device)
                    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    val_loss += outputs.loss.item()

            avg_val = val_loss / max(len(val_loader), 1)
            logger.info(f"          val  loss: {avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), out_weights)
                logger.info(f"  ↳ Saved best model to {out_weights}")

        # Save resumable checkpoint after every epoch
        ckpt_data = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(ckpt_data, checkpoint_path)
        # Keep emergency state up to date so a mid-epoch crash saves the last good epoch
        _emergency_state = {"checkpoint_path": checkpoint_path, "data": ckpt_data}

    if not val_loader:
        torch.save(model.state_dict(), out_weights)
        logger.info(f"Saved final model to {out_weights}")

    logger.info("DETR training complete.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    try:
        train(cfg)
    except (KeyboardInterrupt, Exception) as exc:
        import signal, sys
        # Try to save an emergency checkpoint
        try:
            # _emergency_state is set inside train() via a closure below
            if _emergency_state:
                epath = Path(_emergency_state["checkpoint_path"])
                epath.parent.mkdir(parents=True, exist_ok=True)
                torch.save(_emergency_state["data"], epath)
                logger.info(f"Emergency checkpoint saved to {epath} (epoch {_emergency_state['data']['epoch']})")
        except Exception:
            pass
        if isinstance(exc, KeyboardInterrupt):
            logger.info("Training interrupted by user.")
            sys.exit(0)
        raise
