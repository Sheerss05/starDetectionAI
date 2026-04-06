"""
star_extraction.py
───────────────────
Step 2 — Star Extraction (Non-Trainable)

Uses classical blob-detection (Laplacian of Gaussian or Difference of Gaussian)
to localise bright star-like point sources in a preprocessed sky image.

Exported types
--------------
Star       — named tuple (x, y, sigma, brightness)
StarExtractor — main class
extract_stars — convenience function
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from skimage.feature import blob_log, blob_dog
except Exception:
    blob_log = None
    blob_dog = None


# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Star:
    """
    Represents a single detected star.

    Attributes
    ----------
    x, y       : float — pixel coordinates (column, row)
    sigma      : float — blob scale (roughly proportional to apparent radius)
    brightness : float — normalised peak brightness [0, 1]
    """
    x: float
    y: float
    sigma: float
    brightness: float

    def as_tuple(self) -> Tuple[float, float]:
        """Return (x, y) pixel coordinate."""
        return (self.x, self.y)

    def __repr__(self) -> str:
        return f"Star(x={self.x:.1f}, y={self.y:.1f}, σ={self.sigma:.2f}, b={self.brightness:.3f})"


# ──────────────────────────────────────────────────────────────────────────────
# Extractor
# ──────────────────────────────────────────────────────────────────────────────

class StarExtractor:
    """
    Blob-based star detection.

    Parameters
    ----------
    method : str
        "log" — Laplacian of Gaussian  (more precise, slower)
        "dog" — Difference of Gaussian (faster approximation)
    min_sigma, max_sigma : float
        Scale-space bounds for blob sizes.
    num_sigma : int
        Number of sigma steps between min and max.
    threshold : float
        Blob detection sensitivity (lower = more detections).
    overlap : float
        Max overlap ratio before two blobs are merged [0, 1].
    min_brightness : float
        Minimum normalised pixel brightness to retain a detection [0, 1].
    """

    def __init__(
        self,
        method: str = "log",
        min_sigma: float = 1.0,
        max_sigma: float = 5.0,
        num_sigma: int = 10,
        threshold: float = 0.05,
        overlap: float = 0.5,
        min_brightness: float = 0.15,
    ) -> None:
        self.method = method.lower()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.min_brightness = min_brightness

        if self.method not in ("log", "dog"):
            raise ValueError(f"method must be 'log' or 'dog', got '{method}'")

    # ── main entry ────────────────────────────────────────────────────────────

    def extract(self, image: np.ndarray) -> List[Star]:
        """
        Detect stars in a preprocessed image.

        Parameters
        ----------
        image : np.ndarray
            Float32 array shaped (H, W) or (H, W, C).
            Values expected in [0, 1].

        Returns
        -------
        List[Star]  sorted by brightness (descending).
        """
        grey = self._to_grey(image)

        # Run blob detection
        blobs = self._detect_blobs(grey)     # shape (N, 3): row, col, sigma

        stars: List[Star] = []
        for row, col, sigma in blobs:
            brightness = float(self._sample_brightness(grey, int(row), int(col), sigma))
            if brightness < self.min_brightness:
                continue
            stars.append(Star(
                x=float(col),
                y=float(row),
                sigma=float(sigma),
                brightness=brightness,
            ))

        # Sort brightest-first
        stars.sort(key=lambda s: s.brightness, reverse=True)
        return stars

    def extract_coordinates(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        Convenience wrapper — returns list of (x, y) tuples only.
        """
        return [s.as_tuple() for s in self.extract(image)]

    # ── internals ─────────────────────────────────────────────────────────────

    def _detect_blobs(self, grey: np.ndarray) -> np.ndarray:
        """Run LoG or DoG and return raw blob array."""
        if blob_log is None or blob_dog is None:
            return self._detect_blobs_numpy(grey)

        kwargs = dict(
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            threshold=self.threshold,
            overlap=self.overlap,
        )
        if self.method == "log":
            blobs = blob_log(grey, num_sigma=self.num_sigma, **kwargs)
        else:  # dog
            blobs = blob_dog(grey, sigma_ratio=1.6, **kwargs)

        if blobs.size == 0:
            return np.empty((0, 3))

        # Convert sigma to actual radius for LoG (σ√2)
        if self.method == "log":
            blobs[:, 2] = blobs[:, 2] * (2 ** 0.5)

        return blobs

    def _detect_blobs_numpy(self, grey: np.ndarray) -> np.ndarray:
        """Simple local-maxima fallback when scikit-image is unavailable."""
        h, w = grey.shape
        if h < 3 or w < 3:
            return np.empty((0, 3), dtype=np.float32)

        # Dynamic threshold keeps only bright candidates.
        thr = max(self.threshold, float(np.quantile(grey, 0.995)))
        center = grey[1:-1, 1:-1]
        mask = center >= thr

        # 8-neighborhood non-maximum suppression.
        neighbors = [
            grey[0:-2, 0:-2], grey[0:-2, 1:-1], grey[0:-2, 2:],
            grey[1:-1, 0:-2],                     grey[1:-1, 2:],
            grey[2:,   0:-2], grey[2:,   1:-1], grey[2:,   2:],
        ]
        for n in neighbors:
            mask &= center >= n

        ys, xs = np.where(mask)
        if ys.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        sigma = np.clip((self.min_sigma + self.max_sigma) / 2.0, 1.0, 6.0)
        blobs = np.column_stack([
            ys.astype(np.float32) + 1.0,
            xs.astype(np.float32) + 1.0,
            np.full(ys.shape[0], sigma, dtype=np.float32),
        ])
        return blobs

    @staticmethod
    def _to_grey(image: np.ndarray) -> np.ndarray:
        """Ensure float32 2-D greyscale."""
        if image.ndim == 3:
            if image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                # Weighted luminance
                image = (
                    0.2126 * image[:, :, 0]
                    + 0.7152 * image[:, :, 1]
                    + 0.0722 * image[:, :, 2]
                )
        return image.astype(np.float32)

    @staticmethod
    def _sample_brightness(
        grey: np.ndarray,
        row: int,
        col: int,
        sigma: float,
    ) -> float:
        """
        Estimate peak brightness around a detected blob centre using a small
        local maximum search within the blob radius.
        """
        r = max(1, int(np.ceil(sigma)))
        h, w = grey.shape
        r0, r1 = max(0, row - r), min(h, row + r + 1)
        c0, c1 = max(0, col - r), min(w, col + r + 1)
        patch = grey[r0:r1, c0:c1]
        if patch.size == 0:
            return 0.0
        return float(patch.max())


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def extract_stars(
    image: np.ndarray,
    cfg: dict | None = None,
) -> List[Star]:
    """
    One-call interface for star extraction.

    Parameters
    ----------
    image : preprocessed float32 image
    cfg   : dict with keys matching StarExtractor params, or None.
    """
    cfg = cfg or {}
    extractor = StarExtractor(
        method=cfg.get("method", "log"),
        min_sigma=cfg.get("min_sigma", 1.0),
        max_sigma=cfg.get("max_sigma", 5.0),
        num_sigma=cfg.get("num_sigma", 10),
        threshold=cfg.get("threshold", 0.05),
        overlap=cfg.get("overlap", 0.5),
        min_brightness=cfg.get("min_star_brightness", 0.15),
    )
    return extractor.extract(image)
