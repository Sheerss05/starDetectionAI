"""
preprocessing.py
─────────────────
Step 1 — Image Preprocessing

Converts raw night-sky images into a normalised, noise-reduced representation
that all downstream detectors can consume.

Pipeline:
  1. Load image (RGB → greyscale optional)
  2. Resize to model input size while preserving aspect ratio (letterbox)
  3. CLAHE contrast enhancement   → improves faint-star visibility
  4. Gaussian blur                → reduces pixel-level noise
  5. Normalise to [0, 1] float32  → stable inputs for neural models
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Union, Tuple
from PIL import Image, ImageEnhance, ImageFilter

try:
    import cv2
except Exception:
    cv2 = None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class Preprocessor:
    """
    Encapsulates all image preparation steps.

    Parameters
    ----------
    target_size : (int, int)
        (height, width) to resize the image to after preprocessing.
    gaussian_kernel : int
        Odd kernel size for Gaussian blur.  Set to 0 or 1 to skip.
    clahe_clip : float
        Clip limit for CLAHE.  Higher values → more aggressive enhancement.
    clahe_tile : (int, int)
        Tile grid size for CLAHE.
    keep_rgb : bool
        When True the returned tensor is 3-channel (RGB).
        When False a single greyscale channel is returned.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        gaussian_kernel: int = 3,
        clahe_clip: float = 2.0,
        clahe_tile: Tuple[int, int] = (8, 8),
        keep_rgb: bool = True,
    ) -> None:
        self.target_size = target_size          # (H, W)
        self.gaussian_kernel = gaussian_kernel
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.clahe = (
            cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
            if cv2 is not None else None
        )
        self.keep_rgb = keep_rgb

    # ── main entry point ──────────────────────────────────────────────────────

    def process(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess one image.

        Parameters
        ----------
        source : path-like or numpy array (H, W, C) uint8

        Returns
        -------
        processed : np.ndarray float32
            Shape (H, W, 3) or (H, W, 1) depending on keep_rgb, values [0,1].
        original_rgb : np.ndarray uint8
            The original image resized to target_size — used for visualisation.
        """
        if cv2 is None:
            return self._process_no_cv2(source)

        img = self._load(source)                       # (H, W, 3) uint8 BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Letterbox-resize preserving aspect ratio
        img_resized, original_rgb = self._letterbox(img_rgb, self.target_size)

        # 2. Greyscale copy for enhancement steps
        grey = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)  # (H, W) uint8

        # 3. CLAHE contrast enhancement on grey channel
        grey_clahe = self.clahe.apply(grey)

        # 4. Gaussian noise reduction
        if self.gaussian_kernel > 1:
            grey_clahe = cv2.GaussianBlur(
                grey_clahe,
                (self.gaussian_kernel, self.gaussian_kernel),
                sigmaX=0,
            )

        # 5. Normalise → float32 [0, 1]
        grey_norm = grey_clahe.astype(np.float32) / 255.0      # (H, W)

        if self.keep_rgb:
            # Merge enhanced luminance back into RGB (preserves colour for YOLO/DETR)
            img_lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
            l_enhanced = np.clip(grey_clahe, 0, 255).astype(np.uint8)
            img_lab[:, :, 0] = l_enhanced
            img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

            if self.gaussian_kernel > 1:
                img_enhanced = cv2.GaussianBlur(
                    img_enhanced,
                    (self.gaussian_kernel, self.gaussian_kernel),
                    sigmaX=0,
                )

            processed = img_enhanced.astype(np.float32) / 255.0  # (H, W, 3)
        else:
            processed = grey_norm[:, :, np.newaxis]               # (H, W, 1)

        return processed, original_rgb

    def _process_no_cv2(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback preprocessing using PIL + NumPy when cv2 is unavailable."""
        img_rgb = self._load_no_cv2(source)
        img_resized, original_rgb = self._letterbox_no_cv2(img_rgb, self.target_size)

        pil_img = Image.fromarray(img_resized)
        grey_img = pil_img.convert("L")

        if self.clahe_clip > 0:
            # Approximate local contrast enhancement without OpenCV/skimage.
            contrast_factor = 1.0 + min(3.0, self.clahe_clip / 2.0)
            grey_img = ImageEnhance.Contrast(grey_img).enhance(contrast_factor)

        if self.gaussian_kernel > 1:
            radius = max(0.1, self.gaussian_kernel / 3.0)
            grey_img = grey_img.filter(ImageFilter.GaussianBlur(radius=radius))

        grey_u8 = np.array(grey_img, dtype=np.uint8)
        grey_norm = grey_u8.astype(np.float32) / 255.0

        if self.keep_rgb:
            img_enhanced = np.array(pil_img, dtype=np.uint8)
            if self.gaussian_kernel > 1:
                radius = max(0.1, self.gaussian_kernel / 3.0)
                img_enhanced = np.array(
                    Image.fromarray(img_enhanced).filter(ImageFilter.GaussianBlur(radius=radius)),
                    dtype=np.uint8,
                )

            processed = img_enhanced.astype(np.float32) / 255.0
        else:
            processed = grey_norm[:, :, np.newaxis]

        return processed, original_rgb

    def process_batch(
        self,
        sources: list[Union[str, Path, np.ndarray]],
    ) -> Tuple[np.ndarray, list[np.ndarray]]:
        """
        Preprocess a list of images.

        Returns
        -------
        batch : np.ndarray  (N, H, W, C)
        originals : list of np.ndarray
        """
        results = [self.process(s) for s in sources]
        processed = np.stack([r[0] for r in results], axis=0)
        originals = [r[1] for r in results]
        return processed, originals

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load(source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image from path or pass through numpy array."""
        if isinstance(source, np.ndarray):
            img = source.copy()
            if img.ndim == 2:                       # greyscale → BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:                 # RGBA → BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            return img
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2.imread failed for: {path}")
        return img

    @staticmethod
    def _load_no_cv2(source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image as RGB uint8 without using cv2."""
        if isinstance(source, np.ndarray):
            img = source.copy()
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            return img.astype(np.uint8)
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _letterbox(
        img: np.ndarray,
        target_size: Tuple[int, int],
        fill_color: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize while keeping aspect ratio, pad to target_size.

        Returns
        -------
        padded, resized_only (without padding, for vis reference)
        """
        th, tw = target_size
        h, w = img.shape[:2]
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_top = (th - new_h) // 2
        pad_bottom = th - new_h - pad_top
        pad_left = (tw - new_w) // 2
        pad_right = tw - new_w - pad_left

        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(fill_color, fill_color, fill_color),
        )
        return padded, resized

    @staticmethod
    def _letterbox_no_cv2(
        img: np.ndarray,
        target_size: Tuple[int, int],
        fill_color: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize with aspect ratio and pad using PIL fallback."""
        th, tw = target_size
        h, w = img.shape[:2]
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = np.array(
            Image.fromarray(img.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

        pad_top = (th - new_h) // 2
        pad_bottom = th - new_h - pad_top
        pad_left = (tw - new_w) // 2
        pad_right = tw - new_w - pad_left

        padded = np.pad(
            resized,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=fill_color,
        )
        return padded, resized


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(
    source: Union[str, Path, np.ndarray],
    cfg: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper for single-image preprocessing using a config dict.

    Parameters
    ----------
    source : image path or numpy array
    cfg    : dict with keys matching Preprocessor __init__ params, or None.

    Returns
    -------
    processed, original_rgb
    """
    cfg = cfg or {}
    preprocessor = Preprocessor(
        target_size=tuple(cfg.get("target_size", (640, 640))),
        gaussian_kernel=cfg.get("gaussian_blur_kernel", 3),
        clahe_clip=cfg.get("clahe_clip_limit", 2.0),
        clahe_tile=tuple(cfg.get("clahe_tile_grid", (8, 8))),
    )
    return preprocessor.process(source)
