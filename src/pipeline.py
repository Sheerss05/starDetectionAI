"""
pipeline.py
────────────
Main orchestration module — Step 1 through Step 6.

ConstellationPipeline.run(source) executes the full detection flow:

  1. Preprocessing
  2. Star Extraction
  3. YOLO Detection
  4. DETR Detection
  5. RCNN Detection
  6. Result Fusion

Returns a PipelineResult containing the final Detection list plus
intermediate artefacts for visualisation and debugging.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from src.preprocessing     import Preprocessor
from src.star_extraction   import StarExtractor, Star
from src.yolo_detector     import YOLODetector, Detection
from src.detr_detector     import DetrDetector
from src.rcnn_detector     import RCNNDetector
from src.fusion            import DetectionFusion
from src.visualizer        import ResultVisualizer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """All outputs of a single pipeline run."""

    # ── Primary output ────────────────────────────────────────────────────────
    detections: List[Detection] = field(default_factory=list)

    # ── Intermediate artefacts ────────────────────────────────────────────────
    preprocessed_image: Optional[np.ndarray] = None
    original_rgb: Optional[np.ndarray] = None
    stars: List[Star] = field(default_factory=list)
    yolo_raw: List[Detection] = field(default_factory=list)
    detr_raw: List[Detection] = field(default_factory=list)
    rcnn_raw: List[Detection] = field(default_factory=list)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    elapsed_seconds: float = 0.0
    model_times: Dict[str, float] = field(default_factory=dict)
    image_path: Optional[str] = None

    # ── Public helpers ────────────────────────────────────────────────────────

    def to_json(self) -> List[Dict]:
        """Serialisable output list."""
        return [d.to_dict() for d in self.detections]

    def summarise(self) -> str:
        lines = [
            f"Image        : {self.image_path or 'N/A'}",
            f"Stars found  : {len(self.stars)}",
            f"YOLO raw     : {len(self.yolo_raw)}",
            f"DETR raw     : {len(self.detr_raw)}",
            f"RCNN raw     : {len(self.rcnn_raw)}",
            f"Final dets   : {len(self.detections)}",
            f"Time (s)     : {self.elapsed_seconds:.2f}",
            f"YOLO (ms)    : {self.model_times.get('yolo', 0.0) * 1000:.1f}",
            f"DETR (ms)    : {self.model_times.get('detr', 0.0) * 1000:.1f}",
            f"RCNN (ms)    : {self.model_times.get('rcnn', 0.0) * 1000:.1f}",
            "",
            "Detections:",
        ]
        for d in self.detections:
            lines.append(
                f"  {d.label:<14} conf={d.confidence:.3f}  "
                f"source={d.source}  bbox={[round(v,1) for v in d.bbox]}"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class ConstellationPipeline:
    """
    Full hybrid constellation detection pipeline.

    Parameters
    ----------
    config_path : str | Path | None
        Path to configs/config.yaml.  When None, hardcoded defaults are used.
    device : str
        Override device for all neural models ("cuda" | "cpu").
    class_names : list[str] | None
        Ordered constellation class names (must match training).
    """

    def __init__(
        self,
        config_path: Union[str, Path, None] = "configs/config.yaml",
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        cfg = self._load_config(config_path)
        self.cfg = cfg
        self.class_names = class_names or cfg.get("constellations", [])

        # device override
        if device:
            for section in ("yolo", "detr", "rcnn"):
                if section in cfg:
                    cfg[section]["device"] = device

        # ── Instantiate all components ────────────────────────────────────────
        logger.info("Initialising preprocessing …")
        self.preprocessor = Preprocessor(
            **self._preproc_kwargs(cfg)
        )

        logger.info("Initialising star extractor …")
        self.star_extractor = StarExtractor(
            **self._star_kwargs(cfg)
        )

        logger.info("Initialising YOLO detector …")
        self.yolo = YOLODetector(
            class_names=self.class_names,
            **self._yolo_kwargs(cfg),
        )

        logger.info("Initialising DETR detector …")
        self.detr = DetrDetector(
            class_names=self.class_names,
            **self._detr_kwargs(cfg),
        )

        logger.info("Initialising RCNN detector …")
        self.rcnn = RCNNDetector(
            class_names=self.class_names,
            **self._rcnn_kwargs(cfg),
        )

        logger.info("Initialising Fusion engine …")
        self.fusion = DetectionFusion(
            **self._fusion_kwargs(cfg)
        )

        self.visualizer = ResultVisualizer(
            show_stars=True, show_graph=False
        )

        logger.info("Pipeline ready.")

    # ── main run method ────────────────────────────────────────────────────────

    def run(
        self,
        source: Union[str, Path, np.ndarray],
        return_visuals: bool = False,
        use_yolo: bool = True,
        use_detr: bool = True,
        use_rcnn: bool = True,
        use_gnn: bool = False,  # kept for backward compat; does nothing
        use_fusion: bool = True,
    ) -> PipelineResult:
        """
        Execute the full detection pipeline on one image.

        Parameters
        ----------
        source        : image path or numpy array
        return_visuals: if True, annotated image is stored on the result
        use_yolo      : run the YOLO detector (default True)
        use_detr      : run the DETR detector (default True)
        use_rcnn      : run the RCNN detector (default True)

        Returns
        -------
        PipelineResult
        """
        t0 = time.perf_counter()
        result = PipelineResult(
            image_path=str(source) if not isinstance(source, np.ndarray) else None
        )

        # ── Step 1 — Preprocessing ────────────────────────────────────────────
        logger.debug("Step 1: Preprocessing")
        preprocessed, original_rgb = self.preprocessor.process(source)
        result.preprocessed_image = preprocessed
        result.original_rgb = original_rgb

        # ── Step 2 — Star Extraction ──────────────────────────────────────────
        logger.debug("Step 2: Star extraction")
        stars = self.star_extractor.extract(preprocessed)
        result.stars = stars
        logger.debug(f"  Found {len(stars)} stars.")

        # ── Step 3 — YOLO Detection ───────────────────────────────────────────
        # NOTE: YOLO must receive the original (non-CLAHE) RGB image so that
        # the input distribution matches what the model was trained on.
        # The CLAHE-enhanced `preprocessed` image is used only for star extraction.
        if use_yolo:
            logger.debug("Step 3: YOLO detection")
            _t_yolo = time.perf_counter()
            yolo_dets = self.yolo.detect(original_rgb)
            result.model_times["yolo"] = time.perf_counter() - _t_yolo
            result.yolo_raw = yolo_dets
            logger.debug(f"  YOLO: {len(yolo_dets)} candidate(s)")
        else:
            logger.debug("Step 3: YOLO detection skipped (disabled)")
            result.model_times["yolo"] = 0.0
            yolo_dets = []

        # ── Step 4 — DETR Detection ───────────────────────────────────────────
        # Same reason as YOLO: use original_rgb instead of CLAHE-preprocessed.
        if use_detr:
            logger.debug("Step 4: DETR detection")
            _t_detr = time.perf_counter()
            detr_dets = self.detr.detect(original_rgb)
            result.model_times["detr"] = time.perf_counter() - _t_detr
            result.detr_raw = detr_dets
            logger.debug(f"  DETR: {len(detr_dets)} candidate(s)")
        else:
            logger.debug("Step 4: DETR detection skipped (disabled)")
            result.model_times["detr"] = 0.0
            detr_dets = []

        # ── Step 5 — RCNN Detection ───────────────────────────────────────────
        if use_rcnn:
            logger.debug("Step 5: RCNN detection")
            _t_rcnn = time.perf_counter()
            rcnn_dets = self.rcnn.detect(original_rgb)
            result.model_times["rcnn"] = time.perf_counter() - _t_rcnn
            result.rcnn_raw = rcnn_dets
            logger.debug(f"  RCNN: {len(rcnn_dets)} candidate(s)")
        else:
            logger.debug("Step 5: RCNN detection skipped (disabled)")
            result.model_times["rcnn"] = 0.0
            rcnn_dets = []

        # ── Step 6 — Result Fusion ────────────────────────────────────────────
        logger.debug("Step 6: Fusion")
        active_detectors = int(use_yolo) + int(use_detr) + int(use_rcnn)
        if not use_fusion or active_detectors <= 1:
            # Fusion disabled or only one detector active — keep best confidence per label.
            all_dets = list(yolo_dets + detr_dets + rcnn_dets)
            all_dets.sort(key=lambda d: d.confidence, reverse=True)
            seen: set = set()
            final_dets = []
            for d in all_dets:
                if d.label not in seen:
                    seen.add(d.label)
                    final_dets.append(d)
        else:
            final_dets = self.fusion.fuse(yolo_dets, detr_dets, rcnn_dets)
        result.detections = final_dets

        result.elapsed_seconds = time.perf_counter() - t0
        logger.info(f"Pipeline complete — {len(final_dets)} constellation(s) detected "
                    f"in {result.elapsed_seconds:.2f}s")

        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def visualise(
        self,
        result: PipelineResult,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Render annotated image from a pipeline result.

        Parameters
        ----------
        result    : PipelineResult from .run()
        save_path : optional output path for the annotated image

        Returns
        -------
        Annotated RGB uint8 image.
        """
        base = result.original_rgb
        if base is None:
            raise ValueError("PipelineResult has no original_rgb for visualisation.")

        annotated = self.visualizer.render(
            image=base,
            detections=result.detections,
            stars=result.stars,
        )

        if save_path:
            self.visualizer.save(annotated, save_path)
            logger.info(f"Saved annotated result to {save_path}")

        return annotated

    # ── config parsing ────────────────────────────────────────────────────────

    @staticmethod
    def _load_config(config_path) -> dict:
        if config_path is None:
            return {}
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config not found at {path}, using defaults.")
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _preproc_kwargs(cfg: dict) -> dict:
        pc = cfg.get("preprocessing", {})
        return dict(
            target_size=tuple(pc.get("target_size", (640, 640))),
            gaussian_kernel=pc.get("gaussian_blur_kernel", 3),
            clahe_clip=pc.get("clahe_clip_limit", 2.0),
            clahe_tile=tuple(pc.get("clahe_tile_grid", (8, 8))),
        )

    @staticmethod
    def _star_kwargs(cfg: dict) -> dict:
        sc = cfg.get("star_extraction", {})
        return dict(
            method=sc.get("method", "log"),
            min_sigma=sc.get("min_sigma", 1.0),
            max_sigma=sc.get("max_sigma", 5.0),
            num_sigma=sc.get("num_sigma", 10),
            threshold=sc.get("threshold", 0.05),
            overlap=sc.get("overlap", 0.5),
            min_brightness=sc.get("min_star_brightness", 0.15),
        )

    @staticmethod
    def _yolo_kwargs(cfg: dict) -> dict:
        yc = cfg.get("yolo", {})
        return dict(
            model_weights=yc.get("model_weights", "models/yolo/constellation_yolo.pt"),
            pretrained_base=yc.get("pretrained_base", "yolov8m.pt"),
            img_size=yc.get("img_size", 640),
            conf_threshold=yc.get("conf_threshold", 0.30),
            iou_threshold=yc.get("iou_threshold", 0.45),
            device=yc.get("device", "cuda"),
            num_classes=yc.get("num_classes", 88),
        )

    @staticmethod
    def _detr_kwargs(cfg: dict) -> dict:
        dc = cfg.get("detr", {})
        kwargs = dict(
            model_weights=dc.get("model_weights", "models/detr/constellation_detr.pt"),
            pretrained_base=dc.get("pretrained_base", "facebook/detr-resnet-50"),
            img_size=dc.get("img_size", 800),
            conf_threshold=dc.get("conf_threshold", 0.30),
            num_queries=dc.get("num_queries", 100),
            device=dc.get("device", "cuda"),
            num_classes=dc.get("num_classes", 88),
        )
        if dc.get("annotation_file"):
            kwargs["annotation_file"] = dc["annotation_file"]
        return kwargs

    @staticmethod
    def _rcnn_kwargs(cfg: dict) -> dict:
        rc = cfg.get("rcnn", {})
        kwargs = dict(
            model_weights=rc.get("model_weights", "models/rcnn/constellation_rcnn.pt"),
            pretrained_base=rc.get("pretrained_base", "fasterrcnn_resnet50_fpn"),
            img_size=rc.get("img_size", 800),
            conf_threshold=rc.get("conf_threshold", 0.30),
            iou_threshold=rc.get("iou_threshold", 0.45),
            device=rc.get("device", "cpu"),
            num_classes=rc.get("num_classes", 88),
        )
        if rc.get("annotation_file"):
            kwargs["annotation_file"] = rc["annotation_file"]
        return kwargs

    @staticmethod
    def _fusion_kwargs(cfg: dict) -> dict:
        fc = cfg.get("fusion", {})
        return dict(
            min_model_agreement=fc.get("min_model_agreement", 2),
            iou_merge_threshold=fc.get("iou_merge_threshold", 0.40),
            gnn_override_threshold=fc.get("gnn_override_threshold", 0.85),
            geometry_threshold=fc.get("geometry_threshold", 0.60),
        )
