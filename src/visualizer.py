"""
visualizer.py
──────────────
Visualisation utilities for the constellation detection pipeline.

Provides:
  • draw_detections()  — bounding boxes + labels on the output image
  • draw_stars()       — overlay star positions
  • draw_graph()       — overlay the star connectivity graph
  • save_result()      — write annotated image to disk
  • ResultVisualizer   — class combining all of the above
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.yolo_detector import Detection
from src.star_extraction import Star


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (BGR)
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    (0,   200, 255),  # amber-yellow
    (0,   255, 100),  # spring green
    (255, 100,   0),  # sky blue
    (200,   0, 255),  # magenta
    (0,   150, 255),  # orange
    (100, 255, 255),  # light cyan
    (255, 255,   0),  # cyan
    (50,  255,  50),  # bright green
]

_STAR_COLOUR   = (255, 255, 200)   # pale yellow
_EDGE_COLOUR   = (100, 100, 255)   # dull blue
_VERIFIED_TINT = (0, 220, 80)      # GNN-verified box tint


def _get_colour(idx: int) -> Tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


# ──────────────────────────────────────────────────────────────────────────────
# Drawing primitives
# ──────────────────────────────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    font_scale: float = 0.55,
    thickness: int = 2,
    alpha: float = 0.15,
) -> np.ndarray:
    """
    Draw bounding boxes and constellation labels on a copy of the image.

    Parameters
    ----------
    image      : uint8 BGR or RGB numpy array
    detections : list of Detection results
    font_scale : OpenCV font scale
    thickness  : line thickness in pixels
    alpha      : fill transparency for bounding box rectangle

    Returns
    -------
    Annotated uint8 array (same colour space as input).
    """
    img = image.copy()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    overlay = img.copy()

    label_set = list({d.label for d in detections})

    for i, det in enumerate(detections):
        colour = _get_colour(label_set.index(det.label))
        if det.verified_by_gnn:
            colour = _VERIFIED_TINT

        x1, y1, x2, y2 = [int(v) for v in det.bbox]

        # Semi-transparent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)

        # Solid border
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)

        # Label background
        label_str = (
            f"{det.label}  {det.confidence:.2f}"
            + (" ✓GNN" if det.verified_by_gnn else "")
        )
        (tw, th), baseline = cv2.getTextSize(
            label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        top = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(img, (x1, top), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            img, label_str,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (0, 0, 0), thickness, cv2.LINE_AA,
        )

    # Blend overlay
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def draw_stars(
    image: np.ndarray,
    stars: List[Star],
    radius: int = 3,
    colour: Tuple[int, int, int] = _STAR_COLOUR,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw small circles at each star position.
    """
    img = image.copy()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    for star in stars:
        cx, cy = int(star.x), int(star.y)
        r = max(1, int(star.sigma * 1.5))
        cv2.circle(img, (cx, cy), r, colour, thickness, cv2.LINE_AA)

    return img


def draw_graph(
    image: np.ndarray,
    stars: List[Star],
    adjacency_list: List[Tuple[int, int]],
    node_colour: Tuple[int, int, int] = _STAR_COLOUR,
    edge_colour: Tuple[int, int, int] = _EDGE_COLOUR,
    node_radius: int = 4,
    edge_thickness: int = 1,
) -> np.ndarray:
    """
    Draw the star connectivity graph overlaid on the image.

    Parameters
    ----------
    adjacency_list : list of (src_idx, dst_idx) index pairs into ``stars``.
    """
    img = image.copy()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # Draw edges first (below nodes)
    seen = set()
    for src, dst in adjacency_list:
        key = (min(src, dst), max(src, dst))
        if key in seen:
            continue
        seen.add(key)
        p1 = (int(stars[src].x), int(stars[src].y))
        p2 = (int(stars[dst].x), int(stars[dst].y))
        cv2.line(img, p1, p2, edge_colour, edge_thickness, cv2.LINE_AA)

    # Draw nodes
    for star in stars:
        cv2.circle(
            img, (int(star.x), int(star.y)),
            node_radius, node_colour, -1, cv2.LINE_AA,
        )

    return img


# ──────────────────────────────────────────────────────────────────────────────
# High-level visualiser class
# ──────────────────────────────────────────────────────────────────────────────

class ResultVisualizer:
    """
    Combines detection drawing + optional star / graph overlays.

    Usage
    -----
    vis = ResultVisualizer(show_stars=True, show_graph=False)
    annotated = vis.render(
        original_rgb, detections, stars=stars, adjacency_list=edges
    )
    vis.save(annotated, "output/result.jpg")
    """

    def __init__(
        self,
        show_stars: bool = True,
        show_graph: bool = False,
        font_scale: float = 0.55,
        box_thickness: int = 2,
    ) -> None:
        self.show_stars = show_stars
        self.show_graph = show_graph
        self.font_scale = font_scale
        self.box_thickness = box_thickness

    def render(
        self,
        image: np.ndarray,
        detections: List[Detection],
        stars: Optional[List[Star]] = None,
        adjacency_list: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Produce a fully annotated image.

        Parameters
        ----------
        image       : original image (RGB or BGR, uint8 or float32)
        detections  : fused detection results
        stars       : extracted star list (optional, for overlay)
        adjacency_list : (src, dst) edge index pairs (optional)

        Returns
        -------
        Annotated uint8 array in the same colour space as input.
        """
        img = image.copy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Star overlay
        if self.show_stars and stars:
            img = draw_stars(img, stars)

        # Graph overlay
        if self.show_graph and stars and adjacency_list:
            img = draw_graph(img, stars, adjacency_list)

        # Detection boxes + labels
        img = draw_detections(
            img, detections,
            font_scale=self.font_scale,
            thickness=self.box_thickness,
        )

        return img

    @staticmethod
    def save(image: np.ndarray, path: str | Path, convert_to_bgr: bool = True) -> None:
        """
        Save an annotated image to disk.

        Parameters
        ----------
        image         : RGB uint8 array
        path          : output file path
        convert_to_bgr: set False if image is already in BGR order
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if convert_to_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(out_path), image)

    @staticmethod
    def show(image: np.ndarray, window_title: str = "Constellation Detection") -> None:
        """Display image in an OpenCV window (blocks until key press)."""
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else image
        cv2.imshow(window_title, bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
