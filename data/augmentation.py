"""
augmentation.py
────────────────
Image augmentation utilities for generating training variety.

Supports rotated sky images, brightness/noise variation, and scaling —
critical for training a rotation-invariant constellation detector.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _rotate_box(
    bbox: List[float],
    angle_deg: float,
    cx: float,
    cy: float,
    W: int,
    H: int,
) -> List[float]:
    """Rotate bounding box corners and return a new axis-aligned bbox."""
    x1, y1, x2, y2 = bbox
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    M = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    shifted = corners - np.array([cx, cy])
    rotated = (M @ shifted.T).T + np.array([cx, cy])
    x_min = float(np.clip(rotated[:, 0].min(), 0, W))
    y_min = float(np.clip(rotated[:, 1].min(), 0, H))
    x_max = float(np.clip(rotated[:, 0].max(), 0, W))
    y_max = float(np.clip(rotated[:, 1].max(), 0, H))
    return [x_min, y_min, x_max, y_max]


def random_rotation(
    image: np.ndarray,
    boxes: Optional[List[List[float]]] = None,
    max_degrees: float = 180.0,
    angle: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[List[List[float]]]]:
    """Rotate image and bounding boxes by a random angle (full 360° support)."""
    H, W = image.shape[:2]
    cx, cy = W / 2, H / 2
    if angle is None:
        angle = random.uniform(-max_degrees, max_degrees)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if boxes is None:
        return rotated, None
    rotated_boxes = [_rotate_box(b, angle, cx, cy, W, H) for b in boxes]
    return rotated, rotated_boxes


def random_brightness(
    image: np.ndarray,
    factor_range: Tuple[float, float] = (0.3, 1.5),
) -> np.ndarray:
    """Multiply image brightness by a random factor."""
    factor = random.uniform(*factor_range)
    if image.dtype == np.uint8:
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return np.clip(image * factor, 0.0, 1.0)


def add_gaussian_noise(
    image: np.ndarray,
    std_range: Tuple[float, float] = (0.01, 0.05),
) -> np.ndarray:
    """Add zero-mean Gaussian noise."""
    std = random.uniform(*std_range)
    if image.dtype == np.uint8:
        noise = np.random.normal(0, std * 255, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def random_scale(
    image: np.ndarray,
    boxes: Optional[List[List[float]]] = None,
    scale_range: Tuple[float, float] = (0.7, 1.3),
) -> Tuple[np.ndarray, Optional[List[List[float]]]]:
    """Scale image (zoom in/out) and adjust bounding boxes."""
    H, W = image.shape[:2]
    scale = random.uniform(*scale_range)
    new_W, new_H = int(W * scale), int(H * scale)
    scaled = cv2.resize(image, (new_W, new_H))
    canvas = np.zeros_like(image)

    if scale >= 1.0:
        x_off = (new_W - W) // 2
        y_off = (new_H - H) // 2
        canvas = scaled[y_off:y_off + H, x_off:x_off + W]
        box_ox, box_oy = -x_off, -y_off
    else:
        x_off = (W - new_W) // 2
        y_off = (H - new_H) // 2
        canvas[y_off:y_off + new_H, x_off:x_off + new_W] = scaled
        box_ox, box_oy = x_off, y_off

    if boxes is None:
        return canvas, None

    scaled_boxes = [
        [
            float(np.clip(b[0] * scale + box_ox, 0, W)),
            float(np.clip(b[1] * scale + box_oy, 0, H)),
            float(np.clip(b[2] * scale + box_ox, 0, W)),
            float(np.clip(b[3] * scale + box_oy, 0, H)),
        ]
        for b in boxes
    ]
    return canvas, scaled_boxes


def random_flip(
    image: np.ndarray,
    boxes: Optional[List[List[float]]] = None,
    horizontal: bool = True,
    vertical: bool = True,
) -> Tuple[np.ndarray, Optional[List[List[float]]]]:
    """Random horizontal and/or vertical flip."""
    H, W = image.shape[:2]
    flipped = image.copy()
    flipped_boxes = [list(b) for b in boxes] if boxes else None

    if horizontal and random.random() < 0.5:
        flipped = cv2.flip(flipped, 1)
        if flipped_boxes:
            flipped_boxes = [[W - b[2], b[1], W - b[0], b[3]] for b in flipped_boxes]

    if vertical and random.random() < 0.5:
        flipped = cv2.flip(flipped, 0)
        if flipped_boxes:
            flipped_boxes = [[b[0], H - b[3], b[2], H - b[1]] for b in flipped_boxes]

    return flipped, flipped_boxes


class AugmentationPipeline:
    """
    Chainable augmentation pipeline.

    Parameters
    ----------
    use_rotation, use_brightness, use_noise, use_scale, use_flip : bool
        Toggle individual augmentations.
    rotation_max     : maximum rotation angle (degrees each side, so ±rotation_max)
    brightness_range : (min_factor, max_factor) for brightness multiplier
    noise_range      : (min_std, max_std) as fraction of value range
    scale_range      : (min_scale, max_scale) for image zoom
    """

    def __init__(
        self,
        use_rotation: bool = True,
        use_brightness: bool = True,
        use_noise: bool = True,
        use_scale: bool = True,
        use_flip: bool = True,
        rotation_max: float = 180.0,
        brightness_range: Tuple[float, float] = (0.3, 1.5),
        noise_range: Tuple[float, float] = (0.01, 0.05),
        scale_range: Tuple[float, float] = (0.7, 1.3),
    ) -> None:
        self.use_rotation   = use_rotation
        self.use_brightness = use_brightness
        self.use_noise      = use_noise
        self.use_scale      = use_scale
        self.use_flip       = use_flip
        self.rotation_max   = rotation_max
        self.brightness_range = brightness_range
        self.noise_range    = noise_range
        self.scale_range    = scale_range

    def __call__(
        self,
        image: np.ndarray,
        boxes: Optional[List[List[float]]] = None,
    ) -> Tuple[np.ndarray, Optional[List[List[float]]]]:
        """Apply enabled augmentations in sequence."""
        if self.use_rotation:
            image, boxes = random_rotation(image, boxes, self.rotation_max)
        if self.use_scale:
            image, boxes = random_scale(image, boxes, self.scale_range)
        if self.use_flip:
            image, boxes = random_flip(image, boxes)
        if self.use_brightness:
            image = random_brightness(image, self.brightness_range)
        if self.use_noise:
            image = add_gaussian_noise(image, self.noise_range)
        return image, boxes

    @classmethod
    def from_config(cls, cfg: dict) -> "AugmentationPipeline":
        """Build from a training config dict."""
        return cls(
            use_rotation   = cfg.get("augment", True),
            use_brightness = cfg.get("augment", True),
            use_noise      = cfg.get("augment", True),
            use_scale      = cfg.get("augment", True),
            use_flip       = cfg.get("augment", True),
        )
