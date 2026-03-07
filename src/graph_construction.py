"""
graph_construction.py
──────────────────────
Step 5 — Star Graph Construction

Converts a list of detected star coordinates into a geometric graph:
  • Nodes  = stars  (features: normalised x, y, brightness, sigma)
  • Edges  = k-nearest neighbours filtered by max pixel distance

The resulting graph is expressed as PyTorch Geometric ``Data`` objects
ready for consumption by the GNN in the next stage.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from src.star_extraction import Star


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class StarGraphBuilder:
    """
    Builds a spatial star graph from extracted star coordinates.

    Parameters
    ----------
    k_neighbors : int
        Each star connects to its k nearest neighbours.
    max_edge_distance : float
        Pixel distance cap; edges longer than this are dropped.
    normalize_distances : bool
        When True, edge weights are normalised by max_edge_distance.
    image_size : (int, int)
        (height, width) used to normalise node coordinates to [0, 1].
    """

    def __init__(
        self,
        k_neighbors: int = 6,
        max_edge_distance: float = 150.0,
        normalize_distances: bool = True,
        image_size: Tuple[int, int] = (640, 640),
    ) -> None:
        self.k = k_neighbors
        self.max_dist = max_edge_distance
        self.normalize_distances = normalize_distances
        self.image_size = image_size   # (H, W)

    # ── main entry ────────────────────────────────────────────────────────────

    def build(self, stars: List[Star]) -> Optional[Data]:
        """
        Construct a PyG Data object from a list of Star instances.

        Node features (4-D per node):
          [norm_x, norm_y, brightness, sigma_norm]

        Edge features (1-D per edge):
          [distance_norm]

        Returns None when fewer than 2 stars are present.
        """
        if len(stars) < 2:
            return None

        coords = np.array([[s.x, s.y] for s in stars], dtype=np.float32)
        brightness = np.array([s.brightness for s in stars], dtype=np.float32)
        sigmas = np.array([s.sigma for s in stars], dtype=np.float32)

        H, W = self.image_size
        norm_x = coords[:, 0] / W
        norm_y = coords[:, 1] / H
        sigma_norm = sigmas / (max(sigmas.max(), 1.0))

        # Node feature matrix  (N, 4)
        x = np.stack([norm_x, norm_y, brightness, sigma_norm], axis=1)

        # KNN edges
        src_list, dst_list, dist_list = self._knn_edges(coords)

        if len(src_list) == 0:
            return None

        edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )
        edge_attr = torch.tensor(dist_list, dtype=torch.float32).unsqueeze(1)

        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(stars),
        )

    def build_from_coords(
        self,
        coords: List[Tuple[float, float]],
    ) -> Optional[Data]:
        """
        Lightweight variant when only (x, y) tuples are available.
        Brightness and sigma are set to neutral values (0.5, 0.0).
        """
        stars = [Star(x=c[0], y=c[1], sigma=1.0, brightness=0.5) for c in coords]
        return self.build(stars)

    def build_subgraph(
        self,
        stars: List[Star],
        bbox: List[float],
    ) -> Optional[Data]:
        """
        Build a graph restricted to stars within a bounding box.
        Used by the GNN to evaluate candidate constellation regions.

        Parameters
        ----------
        stars : all detected stars in the image
        bbox  : [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        region_stars = [
            s for s in stars
            if x1 <= s.x <= x2 and y1 <= s.y <= y2
        ]
        if len(region_stars) < 2:
            return None
        return self.build(region_stars)

    # ── edge construction ─────────────────────────────────────────────────────

    def _knn_edges(
        self,
        coords: np.ndarray,
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Build directed k-NN edges (both directions → undirected graph).

        Returns
        -------
        src, dst, distance  — parallel lists.
        """
        from scipy.spatial import cKDTree  # fast k-NN

        tree = cKDTree(coords)
        k = min(self.k + 1, len(coords))   # +1 to exclude self
        distances, indices = tree.query(coords, k=k)

        src_list, dst_list, dist_list = [], [], []
        for i, (dists, nbrs) in enumerate(zip(distances, indices)):
            for d, j in zip(dists[1:], nbrs[1:]):   # skip self (index 0)
                if d > self.max_dist:
                    continue
                norm_d = d / self.max_dist if self.normalize_distances else d
                # Undirected: add both directions
                src_list.extend([i, j])
                dst_list.extend([j, i])
                dist_list.extend([norm_d, norm_d])

        return src_list, dst_list, dist_list

    # ── adjacency matrix ──────────────────────────────────────────────────────

    def adjacency_matrix(self, stars: List[Star]) -> np.ndarray:
        """
        Return a dense (N, N) float32 adjacency matrix of edge distances.
        Entry [i, j] = normalised distance if edge exists, else 0.
        """
        N = len(stars)
        adj = np.zeros((N, N), dtype=np.float32)
        coords = np.array([[s.x, s.y] for s in stars], dtype=np.float32)
        src, dst, dists = self._knn_edges(coords)
        for i, j, d in zip(src, dst, dists):
            adj[i, j] = d
        return adj


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def build_star_graph(
    stars: List[Star],
    cfg: dict | None = None,
    image_size: Tuple[int, int] = (640, 640),
) -> Optional[Data]:
    cfg = cfg or {}
    builder = StarGraphBuilder(
        k_neighbors=cfg.get("k_neighbors", 6),
        max_edge_distance=cfg.get("max_edge_distance", 150.0),
        normalize_distances=cfg.get("normalize_distances", True),
        image_size=image_size,
    )
    return builder.build(stars)
