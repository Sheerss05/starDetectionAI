"""
gnn_model.py
─────────────
Step 6 — Graph Neural Network for Geometry Validation

Architecture
------------
  NodeEncoder  : Linear(4  → hidden)
  EdgeEncoder  : Linear(1  → hidden)
  MessagePassing layers (GCNConv / GATConv with edge features)  × num_layers
  Global mean-pool
  Classifier head  : Linear(hidden → num_classes + 1)

The GNN operates on the star sub-graph within each candidate bounding box.
It outputs a per-class probability distribution to:
  1. Verify whether the spatial star pattern matches a known constellation.
  2. Provide an independent geometry confidence score for fusion.

Two operating modes
-------------------
  • CLASSIFY  — identify the most likely constellation label
  • VERIFY    — binary check: does this sub-graph match a given label?
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Network definition
# ──────────────────────────────────────────────────────────────────────────────

class ConstellationGNN(nn.Module):
    """
    GAT-based graph classifier for constellation geometry verification.

    Parameters
    ----------
    in_channels   : node feature dimensionality  (default 4)
    edge_dim      : edge feature dimensionality  (default 1)
    hidden_channels : width of hidden layers
    num_layers    : number of GAT message-passing layers
    num_classes   : number of constellation classes
    dropout       : dropout probability
    heads         : multi-head attention heads per GAT layer
    """

    def __init__(
        self,
        in_channels: int = 4,
        edge_dim: int = 1,
        hidden_channels: int = 128,
        num_layers: int = 4,
        num_classes: int = 88,
        dropout: float = 0.3,
        heads: int = 4,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.num_classes = num_classes

        # ── Node encoder ──────────────────────────────────────────────────────
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )

        # ── Edge encoder ──────────────────────────────────────────────────────
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.ReLU(),
        )

        # ── GAT layers ────────────────────────────────────────────────────────
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # First layer: hidden_channels → hidden_channels (with concat heads)
        self.conv_layers.append(
            GATConv(
                hidden_channels,
                hidden_channels // heads,
                heads=heads,
                edge_dim=hidden_channels,
                dropout=dropout,
                concat=True,
            )
        )
        self.norm_layers.append(nn.LayerNorm(hidden_channels))

        # Subsequent layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                GATConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    edge_dim=hidden_channels,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norm_layers.append(nn.LayerNorm(hidden_channels))

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(self, data: Data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Data object
            data.x          — (N, in_channels)
            data.edge_index — (2, E)
            data.edge_attr  — (E, edge_dim)
            data.batch      — (N,) batch assignment vector

        Returns
        -------
        logits : (batch_size, num_classes)
        """
        x = self.node_encoder(data.x)
        edge_feat = self.edge_encoder(data.edge_attr)

        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x_new = conv(x, data.edge_index, edge_attr=edge_feat)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new if x.shape == x_new.shape else x_new   # residual

        # Global mean pooling → graph-level representation
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
            torch.zeros(data.num_nodes, dtype=torch.long, device=x.device)
        out = global_mean_pool(x, batch)   # (B, hidden)

        return self.classifier(out)        # (B, num_classes)

    # ── convenience helpers ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict_proba(self, data: Data) -> torch.Tensor:
        """Return softmax probability distribution (B, num_classes)."""
        self.eval()
        logits = self(data)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def verify(self, data: Data, class_idx: int) -> float:
        """
        Return the GNN's confidence that graph 'data' represents
        the constellation at index 'class_idx'.
        """
        proba = self.predict_proba(data)
        return float(proba[0, class_idx].item())


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper for inference
# ──────────────────────────────────────────────────────────────────────────────

class GNNValidator:
    """
    Loads a trained ConstellationGNN and verifies candidate detections.

    Parameters
    ----------
    model_weights : str | Path
        Path to saved .pt state dict.
    hidden_channels, num_layers, dropout, num_classes : int / float
        Must match the architecture used during training.
    device : str
    class_names : list[str] | None
    geometry_threshold : float
        Minimum verification score to consider GNN positive.
    """

    def __init__(
        self,
        model_weights: str | Path = "models/gnn/constellation_gnn.pt",
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 88,
        device: str = "cuda",
        class_names: Optional[List[str]] = None,
        geometry_threshold: float = 0.60,
    ) -> None:
        self.device = self._resolve_device(device)
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.geometry_threshold = geometry_threshold
        self.model = self._load_model(
            model_weights, hidden_channels, num_layers, dropout, num_classes
        )

    # ── validate a single detection ───────────────────────────────────────────

    def validate(
        self,
        graph: Data,
        candidate_label: str,
    ) -> tuple[bool, float]:
        """
        Validate whether the star sub-graph matches the candidate label.

        Returns
        -------
        (verified : bool, gnn_score : float)
        """
        if self.model is None or graph is None:
            return False, 0.0

        graph = graph.to(self.device)

        try:
            cls_idx = self.class_names.index(candidate_label)
        except ValueError:
            logger.warning(f"Label '{candidate_label}' not in class_names.")
            return False, 0.0

        score = self.model.verify(graph, cls_idx)
        verified = score >= self.geometry_threshold
        return verified, score

    # ── batch validate multiple detections ────────────────────────────────────

    def validate_batch(
        self,
        graphs: List[Optional[Data]],
        labels: List[str],
    ) -> List[tuple[bool, float]]:
        """Validate a list of (graph, label) pairs."""
        return [
            self.validate(g, l) if g is not None else (False, 0.0)
            for g, l in zip(graphs, labels)
        ]

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(
        self, weights_path, hidden_channels, num_layers, dropout, num_classes
    ) -> Optional[ConstellationGNN]:
        model = ConstellationGNN(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        ).to(self.device)

        weights_path = Path(weights_path)
        if weights_path.exists():
            logger.info(f"Loading GNN weights from {weights_path}")
            state = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state)
        else:
            logger.warning(
                f"GNN weights not found at {weights_path}. "
                "Model is randomly initialised — verification scores will be unreliable."
            )

        model.eval()
        return model

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable — using CPU.")
            return "cpu"
        return device
