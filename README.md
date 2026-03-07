# 🌌 starAIv2 — Constellation Recognition AI

A hybrid deep learning system that detects multiple constellations in a single night-sky image using **YOLO + DETR + Graph Neural Network (GNN)** architecture.

---

## Architecture Overview

```
Input Image
      ↓
Preprocessing          (CLAHE, Gaussian blur, letterbox resize)
      ↓
Star Extraction        (Blob detection — LoG / DoG, non-trainable)
      ↓
 ┌────────────┬──────────────┬─────────────────────────┐
 │            │              │
YOLO         DETR           GNN (Graph Neural Network)
 │            │    ↑             ↑
 │            │    └─── Star graph (k-NN from extracted stars)
 └────────────┴──────────────┘
      ↓
Result Fusion           (Model agreement + GNN geometry validation)
      ↓
Final Constellation Output
```

| Component | Role |
|---|---|
| **Preprocessing** | Normalize brightness, reduce noise, enhance star visibility |
| **Star Extraction** | Detect stars via blob detection (no CNN) |
| **YOLO** | Fast primary region detector |
| **DETR** | Global transformer-based detector, reduces duplicates in dense fields |
| **GNN** | Geometric structure verifier — validates star relationship patterns |
| **Fusion** | Combines all three outputs, applies acceptance rules |

---

## Output Format

```json
{
  "constellation_name": "Orion",
  "bounding_box": [120.0, 80.0, 410.0, 390.0],
  "confidence_score": 0.8750,
  "verified_by_GNN": true,
  "gnn_score": 0.7812,
  "source": "fused"
}
```

---

## Project Structure

```
starAIv2/
├── src/
│   ├── preprocessing.py       # Step 1 — image preprocessing
│   ├── star_extraction.py     # Step 2 — blob-based star detection
│   ├── yolo_detector.py       # Step 3 — YOLO detection
│   ├── detr_detector.py       # Step 4 — DETR detection
│   ├── graph_construction.py  # Step 5 — k-NN star graph builder
│   ├── gnn_model.py           # Step 6 — GAT-based GNN validator
│   ├── fusion.py              # Step 7 — multi-model result fusion
│   ├── visualizer.py          # Annotation drawing utilities
│   └── pipeline.py            # Full orchestration
│
├── training/
│   ├── train_yolo.py          # YOLOv8 fine-tuning
│   ├── train_detr.py          # DETR fine-tuning
│   └── train_gnn.py           # GNN classification training
│
├── data/
│   ├── dataset.py             # COCO↔YOLO conversion, graph builder
│   ├── augmentation.py        # Rotation, brightness, noise, scale, flip
│   └── constellation_dataset/
│       └── dataset.yaml       # YOLOv8 dataset config stub
│
├── models/
│   ├── yolo/                  # YOLO weights (constellation_yolo.pt)
│   ├── detr/                  # DETR weights (constellation_detr.pt)
│   └── gnn/                   # GNN  weights (constellation_gnn.pt)
│
├── configs/
│   └── config.yaml            # Master configuration
│
├── tests/
│   └── test_pipeline.py       # Unit + integration tests
│
├── main.py                    # CLI entry point
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `torch-geometric` may need separate CUDA-specific installation.
> See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Usage

### Inference

```bash
# Detect constellations in an image
python main.py infer --image sky.jpg

# Save annotated result image
python main.py infer --image sky.jpg --save output/result.jpg

# Output JSON detections to stdout
python main.py infer --image sky.jpg --json -

# Force CPU
python main.py infer --image sky.jpg --device cpu
```

### Training

```bash
# Train YOLO
python main.py train --model yolo --config configs/config.yaml

# Train DETR
python main.py train --model detr --config configs/config.yaml

# Train GNN
python main.py train --model gnn --config configs/config.yaml
```

### Dataset Preparation

```bash
# Convert COCO annotations to YOLO format
python main.py convert --direction coco2yolo \
  --coco data/annotations.json --out data/labels/

# Build GNN graph dataset from annotated images
python main.py build-graphs \
  --coco data/annotations.json --images data/images/ \
  --output data/graph_dataset/
```

### Tests

```bash
pytest tests/test_pipeline.py -v
```

---

## Training Data Requirements

### YOLO & DETR
- Bounding box annotations (COCO or YOLO format)
- Multiple constellations per image
- Recommended augmentations (built-in): ±180° rotation, brightness ×0.3–1.5, Gaussian noise, scaling ×0.7–1.3, h/v flip

### GNN
- Star coordinate graphs extracted from image bounding-box regions
- Class label per graph
- Built automatically via:
  ```bash
  python main.py build-graphs --coco annotations.json --images images/
  ```

---

## Fusion Rules

A detection is accepted when **any** of the following hold:

| Rule | Condition |
|---|---|
| **Model Agreement** | ≥ 2 detectors produce overlapping boxes (IoU ≥ 0.40) for the same label |
| **GNN Override** | GNN geometry score ≥ 0.85 (regardless of detector count) |
| **High Confidence** | Single detector confidence > 0.90 AND GNN score ≥ threshold |

---

## Configuration

All hyperparameters are in [`configs/config.yaml`](configs/config.yaml):

| Section | Key parameters |
|---|---|
| `preprocessing` | `target_size`, `clahe_clip_limit`, `gaussian_blur_kernel` |
| `star_extraction` | `method` (log/dog), `threshold`, `min_star_brightness` |
| `yolo` | `conf_threshold`, `iou_threshold`, `model_weights` |
| `detr` | `conf_threshold`, `num_queries`, `model_weights` |
| `gnn` | `hidden_channels`, `num_layers`, `geometry_threshold` |
| `fusion` | `min_model_agreement`, `iou_merge_threshold`, `gnn_override_threshold` |

---

## Performance Goals

- ✅ Detect multiple constellations per image
- ✅ Robust in dense star environments (GNN validates geometry)
- ✅ Supports arbitrary rotation (full ±180° augmentation)
- ✅ Reduces false positives via 3-model fusion
- ✅ Variable brightness and noise tolerance
