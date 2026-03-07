"""
train_gnn.py
─────────────
Train the Graph Neural Network to classify constellation geometric patterns.

Dataset layout
──────────────
  data/graph_dataset/
    graphs/
      <constellation_name>/
        <sample_id>.pt         ← PyG Data object saved with torch.save()
    split.json                  ← {"train": [...], "val": [...]}

A .pt file contains: Data(x, edge_index, edge_attr, y)
  x          : (N, 4) node features  [norm_x, norm_y, brightness, sigma_norm]
  edge_index : (2, E)
  edge_attr  : (E, 1)
  y          : (1,) long  ← class index

Run
───
  python training/train_gnn.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.gnn_model import ConstellationGNN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ConstellationGraphDataset(torch.utils.data.Dataset):
    """
    Loads pre-computed PyG graph Data objects from disk.

    Parameters
    ----------
    root        : base dataset directory
    class_names : ordered list of constellation names (determines class idx)
    split       : "train" | "val"
    """

    def __init__(self, root: str, class_names: List[str], split: str = "train") -> None:
        self.class_names = class_names
        root = Path(root)

        split_file = root / "split.json"
        if split_file.exists():
            with open(split_file) as f:
                split_data = json.load(f)
            self.files: List[Path] = [Path(p) for p in split_data.get(split, [])]
        else:
            # Fall back: scan all .pt files and split 80/20
            all_files = sorted((root / "graphs").rglob("*.pt"))
            cut = int(len(all_files) * 0.8)
            self.files = all_files[:cut] if split == "train" else all_files[cut:]
            logger.warning("split.json not found — using automatic 80/20 split.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        data = torch.load(self.files[idx])
        # Ensure .y is a long scalar tensor
        if not hasattr(data, "y") or data.y is None:
            # Infer label from parent directory name
            label = self.files[idx].parent.name
            cls_idx = self.class_names.index(label) if label in self.class_names else 0
            data.y = torch.tensor([cls_idx], dtype=torch.long)
        return data


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    gnn_cfg       = cfg.get("gnn", {})
    train_cfg     = cfg.get("train_gnn", {})
    class_names   = cfg.get("constellations", [str(i) for i in range(88)])

    device        = gnn_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    epochs        = int(train_cfg.get("epochs", 200))
    batch_size    = int(train_cfg.get("batch_size", 32))
    lr            = float(train_cfg.get("lr", 1e-3))
    weight_decay  = float(train_cfg.get("weight_decay", 5e-4))
    data_root     = train_cfg.get("data_root", "data/graph_dataset")
    num_classes   = int(gnn_cfg.get("num_classes", len(class_names)))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConstellationGNN(
        hidden_channels=gnn_cfg.get("hidden_channels", 128),
        num_layers=gnn_cfg.get("num_layers", 4),
        dropout=gnn_cfg.get("dropout", 0.3),
        num_classes=num_classes,
    ).to(device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    data_root_path = Path(data_root)
    if not data_root_path.exists():
        logger.error(f"Graph dataset not found at {data_root}.  Exiting.")
        return

    train_ds = ConstellationGraphDataset(data_root, class_names, split="train")
    val_ds   = ConstellationGraphDataset(data_root, class_names, split="val")
    logger.info(f"Train: {len(train_ds)} graphs  |  Val: {len(val_ds)} graphs")

    if len(train_ds) == 0:
        logger.error("Training dataset is empty.  Check data/graph_dataset/.")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = torch.nn.CrossEntropyLoss()

    out_weights = Path(gnn_cfg.get("model_weights", "models/gnn/constellation_gnn.pt"))
    out_weights.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=-1)
            correct += (preds == batch.y.view(-1)).sum().item()
            total += batch.num_graphs

        scheduler.step()
        train_acc = correct / max(total, 1)
        avg_loss  = total_loss / max(total, 1)

        # ── val ────────────────────────────────────────────────────────
        if len(val_ds) > 0:
            model.eval()
            vcorrect, vtotal = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    preds = model(batch).argmax(dim=-1)
                    vcorrect += (preds == batch.y.view(-1)).sum().item()
                    vtotal   += batch.num_graphs
            val_acc = vcorrect / max(vtotal, 1)
        else:
            val_acc = train_acc

        logger.info(
            f"Epoch {epoch:3d}  loss={avg_loss:.4f}  "
            f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_weights)
            logger.info(f"  ↳ Saved best model  (val_acc={val_acc:.3f})")

    logger.info(f"GNN training complete.  Best val_acc={best_val_acc:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
