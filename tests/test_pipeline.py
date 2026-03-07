"""
test_pipeline.py
─────────────────
Unit and integration tests for the Constellation Recognition AI.

Tests cover:
  • Preprocessing (shape, value range, letterbox)
  • Star Extraction (blob detection, Star dataclass)
  • Graph Construction (node/edge counts, PyG Data validity)
  • GNN model (forward pass shape, verify method)
  • Detection dataclass (to_dict, area, centre)
  • Result Fusion (IoU helper, fusion rules)
  • Augmentation (rotation, brightness, noise)
  • Pipeline (end-to-end dry run with synthetic image)

Run
───
  pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_sky():
    """
    512×512 synthetic 'night sky' image — black with bright dots for stars.
    Returned as float32 RGB [0, 1].
    """
    rng = np.random.default_rng(42)
    img = np.zeros((512, 512, 3), dtype=np.float32)
    # Add 30 bright star-like points
    for _ in range(30):
        cx, cy = rng.integers(10, 502, size=2)
        r = rng.integers(1, 4)
        brightness = rng.uniform(0.5, 1.0)
        import cv2
        cv2.circle(img, (int(cx), int(cy)), int(r),
                   (brightness, brightness, brightness), -1)
    return img


@pytest.fixture
def synthetic_sky_uint8(synthetic_sky):
    return (synthetic_sky * 255).astype(np.uint8)


@pytest.fixture
def sample_stars():
    """List of 10 synthetic Star objects spread across a 640×640 canvas."""
    from src.star_extraction import Star
    rng = np.random.default_rng(0)
    return [
        Star(x=float(x), y=float(y), sigma=2.0, brightness=0.6)
        for x, y in rng.integers(50, 590, size=(10, 2))
    ]


@pytest.fixture
def sample_detections():
    """A pair of synthetic Detection objects."""
    from src.yolo_detector import Detection
    return [
        Detection(label="Orion",  bbox=[50., 50., 200., 200.], confidence=0.82, source="yolo"),
        Detection(label="Gemini", bbox=[250., 50., 400., 200.], confidence=0.75, source="detr"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    def test_output_shape(self, synthetic_sky_uint8):
        from src.preprocessing import Preprocessor
        pp = Preprocessor(target_size=(640, 640))
        processed, original = pp.process(synthetic_sky_uint8)
        assert processed.shape == (640, 640, 3), "Wrong processed shape"
        assert original.shape[:2] == (512, 512) or original.shape[0] <= 640

    def test_value_range(self, synthetic_sky_uint8):
        from src.preprocessing import Preprocessor
        pp = Preprocessor(target_size=(640, 640))
        processed, _ = pp.process(synthetic_sky_uint8)
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0

    def test_greyscale_output(self, synthetic_sky_uint8):
        from src.preprocessing import Preprocessor
        pp = Preprocessor(target_size=(320, 320), keep_rgb=False)
        processed, _ = pp.process(synthetic_sky_uint8)
        assert processed.shape == (320, 320, 1)

    def test_letterbox_non_square(self):
        """Non-square input should be letterboxed without cropping content."""
        import cv2
        from src.preprocessing import Preprocessor
        img = np.zeros((400, 800, 3), dtype=np.uint8)  # 2:1 aspect ratio
        img[200, 400] = [255, 255, 255]  # single white pixel at centre
        pp = Preprocessor(target_size=(640, 640))
        processed, _ = pp.process(img)
        assert processed.shape == (640, 640, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Star Extraction tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStarExtraction:
    def test_returns_stars(self, synthetic_sky):
        from src.star_extraction import StarExtractor, Star
        extractor = StarExtractor(threshold=0.02, min_brightness=0.1)
        stars = extractor.extract(synthetic_sky)
        assert isinstance(stars, list)
        assert all(isinstance(s, Star) for s in stars)

    def test_detects_bright_points(self, synthetic_sky):
        from src.star_extraction import StarExtractor
        extractor = StarExtractor(threshold=0.02, min_brightness=0.1)
        stars = extractor.extract(synthetic_sky)
        # Should detect most of the 30 planted stars
        assert len(stars) >= 10, f"Too few stars detected: {len(stars)}"

    def test_sorted_by_brightness(self, synthetic_sky):
        from src.star_extraction import StarExtractor
        extractor = StarExtractor(threshold=0.02, min_brightness=0.05)
        stars = extractor.extract(synthetic_sky)
        if len(stars) > 1:
            for a, b in zip(stars, stars[1:]):
                assert a.brightness >= b.brightness

    def test_dog_method(self, synthetic_sky):
        from src.star_extraction import StarExtractor
        extractor = StarExtractor(method="dog", threshold=0.02, min_brightness=0.1)
        stars = extractor.extract(synthetic_sky)
        assert isinstance(stars, list)

    def test_extract_coordinates(self, synthetic_sky):
        from src.star_extraction import StarExtractor
        extractor = StarExtractor(threshold=0.02, min_brightness=0.1)
        coords = extractor.extract_coordinates(synthetic_sky)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in coords)


# ══════════════════════════════════════════════════════════════════════════════
# Graph Construction tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGraphConstruction:
    def test_builds_pyg_data(self, sample_stars):
        from src.graph_construction import StarGraphBuilder
        from torch_geometric.data import Data
        builder = StarGraphBuilder(k_neighbors=4, max_edge_distance=300.0)
        graph = builder.build(sample_stars)
        assert isinstance(graph, Data)

    def test_node_count(self, sample_stars):
        from src.graph_construction import StarGraphBuilder
        builder = StarGraphBuilder(k_neighbors=4, max_edge_distance=300.0)
        graph = builder.build(sample_stars)
        assert graph.num_nodes == len(sample_stars)

    def test_node_features_dim(self, sample_stars):
        from src.graph_construction import StarGraphBuilder
        builder = StarGraphBuilder(k_neighbors=4, max_edge_distance=300.0)
        graph = builder.build(sample_stars)
        assert graph.x.shape[1] == 4

    def test_edge_attr_exists(self, sample_stars):
        from src.graph_construction import StarGraphBuilder
        builder = StarGraphBuilder(k_neighbors=4, max_edge_distance=300.0)
        graph = builder.build(sample_stars)
        assert graph.edge_attr is not None
        assert graph.edge_attr.shape[1] == 1

    def test_edge_distances_normalised(self, sample_stars):
        from src.graph_construction import StarGraphBuilder
        builder = StarGraphBuilder(k_neighbors=4, max_edge_distance=300.0, normalize_distances=True)
        graph = builder.build(sample_stars)
        assert graph.edge_attr.max().item() <= 1.0 + 1e-5

    def test_returns_none_for_single_star(self):
        from src.star_extraction import Star
        from src.graph_construction import StarGraphBuilder
        builder = StarGraphBuilder()
        result = builder.build([Star(100, 100, 2.0, 0.7)])
        assert result is None

    def test_subgraph_restricted_to_bbox(self, sample_stars):
        from src.graph_construction import StarGraphBuilder
        builder = StarGraphBuilder(k_neighbors=4, max_edge_distance=999.0)
        # Use a very wide bbox to catch all stars
        graph_all  = builder.build(sample_stars)
        graph_sub  = builder.build_subgraph(sample_stars, [0, 0, 640, 640])
        assert graph_sub is not None
        assert graph_sub.num_nodes <= graph_all.num_nodes


# ══════════════════════════════════════════════════════════════════════════════
# GNN model tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGNNModel:
    def _make_graph(self, n_nodes=8):
        from torch_geometric.data import Data
        x = torch.rand(n_nodes, 4)
        # Complete graph edges
        src = torch.tensor([i for i in range(n_nodes) for j in range(n_nodes) if i != j])
        dst = torch.tensor([j for i in range(n_nodes) for j in range(n_nodes) if i != j])
        edge_index = torch.stack([src, dst])
        edge_attr = torch.rand(edge_index.shape[1], 1)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_nodes)

    def test_forward_shape(self):
        from src.gnn_model import ConstellationGNN
        model = ConstellationGNN(num_classes=10, hidden_channels=32, num_layers=2)
        model.eval()
        data = self._make_graph()
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        logits = model(data)
        assert logits.shape == (1, 10)

    def test_predict_proba_sums_to_one(self):
        from src.gnn_model import ConstellationGNN
        model = ConstellationGNN(num_classes=5, hidden_channels=32, num_layers=2)
        data = self._make_graph()
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        proba = model.predict_proba(data)
        assert abs(proba.sum().item() - 1.0) < 1e-5

    def test_verify_returns_float(self):
        from src.gnn_model import ConstellationGNN
        model = ConstellationGNN(num_classes=5, hidden_channels=32, num_layers=2)
        data = self._make_graph()
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        score = model.verify(data, class_idx=0)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Detection dataclass tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDetection:
    def test_area(self):
        from src.yolo_detector import Detection
        d = Detection(label="Orion", bbox=[0, 0, 100, 50], confidence=0.9)
        assert d.area == 5000.0

    def test_centre(self):
        from src.yolo_detector import Detection
        d = Detection(label="Orion", bbox=[0, 0, 100, 100], confidence=0.9)
        assert d.centre == (50.0, 50.0)

    def test_to_dict_keys(self):
        from src.yolo_detector import Detection
        d = Detection(label="Orion", bbox=[0, 0, 100, 100], confidence=0.9, verified_by_gnn=True)
        d_dict = d.to_dict()
        for key in ("constellation_name", "bounding_box", "confidence_score",
                    "verified_by_GNN", "gnn_score", "source"):
            assert key in d_dict


# ══════════════════════════════════════════════════════════════════════════════
# Fusion tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFusion:
    def test_iou_identical_boxes(self):
        from src.fusion import _iou
        assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        from src.fusion import _iou
        assert _iou([0, 0, 5, 5], [10, 10, 20, 20]) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        from src.fusion import _iou
        iou = _iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.0 < iou < 1.0

    def test_fusion_agreement_rule(self):
        """Two detectors agreeing on Orion → Orion accepted."""
        from src.yolo_detector import Detection
        from src.fusion import DetectionFusion
        yolo = [Detection("Orion", [0, 0, 100, 100], 0.8, "yolo")]
        detr = [Detection("Orion", [5, 5, 105, 105], 0.75, "detr")]
        gnn_scores = [(yolo[0], 0.5), (detr[0], 0.5)]
        engine = DetectionFusion(min_model_agreement=2, iou_merge_threshold=0.3, geometry_threshold=0.4)
        results = engine.fuse(yolo, detr, gnn_scores)
        assert any(d.label == "Orion" for d in results)

    def test_fusion_rejects_unverified_single(self):
        """Single low-confidence detection with low GNN score → rejected."""
        from src.yolo_detector import Detection
        from src.fusion import DetectionFusion
        yolo = [Detection("Gemini", [0, 0, 100, 100], 0.4, "yolo")]
        engine = DetectionFusion(min_model_agreement=2, gnn_override_threshold=0.85)
        results = engine.fuse(yolo, [], [(yolo[0], 0.3)])
        assert all(d.label != "Gemini" for d in results)

    def test_fusion_gnn_override(self):
        """High GNN score alone should accept regardless of model agreement."""
        from src.yolo_detector import Detection
        from src.fusion import DetectionFusion
        yolo = [Detection("Lyra", [0, 0, 100, 100], 0.55, "yolo")]
        engine = DetectionFusion(min_model_agreement=2, gnn_override_threshold=0.80)
        results = engine.fuse(yolo, [], [(yolo[0], 0.90)])
        assert any(d.label == "Lyra" for d in results)


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAugmentation:
    def test_rotation_preserves_shape(self, synthetic_sky_uint8):
        from data.augmentation import random_rotation
        rotated, _ = random_rotation(synthetic_sky_uint8, angle=45.0)
        assert rotated.shape == synthetic_sky_uint8.shape

    def test_rotation_adjusts_boxes(self, synthetic_sky_uint8):
        from data.augmentation import random_rotation
        H, W = synthetic_sky_uint8.shape[:2]
        boxes = [[10, 10, 100, 100]]
        _, rotated_boxes = random_rotation(synthetic_sky_uint8, boxes, angle=90.0)
        assert rotated_boxes is not None
        assert len(rotated_boxes) == 1
        x1, y1, x2, y2 = rotated_boxes[0]
        assert 0 <= x1 <= W and 0 <= x2 <= W
        assert 0 <= y1 <= H and 0 <= y2 <= H

    def test_brightness_range(self, synthetic_sky):
        from data.augmentation import random_brightness
        out = random_brightness(synthetic_sky, factor_range=(0.5, 0.5))
        assert out.max() <= 1.0 + 1e-5

    def test_noise_shape(self, synthetic_sky):
        from data.augmentation import add_gaussian_noise
        out = add_gaussian_noise(synthetic_sky)
        assert out.shape == synthetic_sky.shape
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5

    def test_pipeline_applies_all(self, synthetic_sky_uint8):
        from data.augmentation import AugmentationPipeline
        aug = AugmentationPipeline()
        out, _ = aug(synthetic_sky_uint8)
        assert out.shape == synthetic_sky_uint8.shape


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end pipeline smoke test (CPU, no trained weights)
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineSmoke:
    """
    Runs the full pipeline without trained weights to verify the integration
    plumbing works — i.e., no exceptions are raised and the result structure
    is correct.

    YOLO and DETR will produce no detections (weights not loaded);
    the result should be an empty detection list.
    """

    def test_pipeline_runs_without_crash(self, synthetic_sky_uint8):
        from src.pipeline import ConstellationPipeline, PipelineResult
        pipeline = ConstellationPipeline(
            config_path=None,  # use defaults
            device="cpu",
        )
        result = pipeline.run(synthetic_sky_uint8)
        assert isinstance(result, PipelineResult)
        assert isinstance(result.detections, list)
        assert isinstance(result.stars, list)
        assert result.elapsed_seconds > 0

    def test_pipeline_result_to_json(self, synthetic_sky_uint8):
        from src.pipeline import ConstellationPipeline
        pipeline = ConstellationPipeline(config_path=None, device="cpu")
        result = pipeline.run(synthetic_sky_uint8)
        output = result.to_json()
        assert isinstance(output, list)

    def test_pipeline_summary_string(self, synthetic_sky_uint8):
        from src.pipeline import ConstellationPipeline
        pipeline = ConstellationPipeline(config_path=None, device="cpu")
        result = pipeline.run(synthetic_sky_uint8)
        summary = result.summarise()
        assert "Stars found" in summary
        assert "Time (s)" in summary
