# 🌌 Star AI — Constellation Recognition System

A hybrid deep learning system that detects multiple constellations in a single night-sky image using a **YOLO + DETR + Faster R-CNN** ensemble with multi-model fusion. It includes both a command-line interface and an interactive Streamlit web application.

---

## Architecture Overview

```
Input Image
      ↓
Preprocessing          (CLAHE, Gaussian blur, letterbox resize)
      ↓
Star Extraction        (Blob detection — LoG / DoG, non-trainable)
      ↓
 ┌────────────┬──────────────┬──────────────────┐
 │            │              │
YOLO         DETR           Faster R-CNN (RCNN)
(YOLOv8)    (ResNet-50)    (ResNet-50-FPN)
 └────────────┴──────────────┴──────────────────┘
      ↓
Result Fusion           (Multi-model agreement + IoU merging)
      ↓
Final Constellation Output
```

| Component | Role |
|---|---|
| **Star Extraction** | Detect stars via blob detection (LoG / DoG) |
| **YOLO** | Fast single-pass primary region detector (YOLOv8m) |
| **DETR** | Global transformer-based detector, handles dense/overlapping fields |
| **Faster R-CNN** | Region-proposal detector, effective for variable-scale and partial overlaps |
| **Fusion** | Combines all three outputs using model agreement and IoU merging |

---

## Supported Constellations

The system is trained on **17 primary classes** with a dataset covering up to 88 IAU constellations:

Orion · Ursa Major · Ursa Minor · Cassiopeia · Cygnus · Leo · Scorpius · Gemini · Taurus · Virgo · Aquila · Perseus · Lyra · Boötes · Pegasus · Sagittarius · Aquarius

---

## Output Format

```json
{
  "constellation_name": "Orion",
  "bounding_box": [120.0, 80.0, 410.0, 390.0],
  "confidence_score": 0.8750,
  "source": "fused"
}
```

---

## Project Structure

```
starAI/
├── app.py                         # Streamlit web application
├── main.py                        # CLI entry point
├── requirements.txt
│
├── src/
│   ├── preprocessing.py           # Step 1 — image preprocessing
│   ├── star_extraction.py         # Step 2 — blob-based star detection
│   ├── yolo_detector.py           # Step 3 — YOLO detection
│   ├── detr_detector.py           # Step 4 — DETR detection
│   ├── rcnn_detector.py           # Step 5 — Faster R-CNN detection
│   ├── fusion.py                  # Step 6 — multi-model result fusion
│   ├── visualizer.py              # Annotation drawing utilities
│   ├── pipeline.py                # Full orchestration
│   ├── graph_construction.py      # k-NN star graph builder (utility)
│   └── gnn_model.py               # GAT-based GNN (experimental)
│
├── training/
│   ├── train_yolo.py              # YOLOv8 fine-tuning
│   ├── train_detr.py              # DETR fine-tuning
│   └── train_gnn.py               # GNN training (experimental)
│
├── data/
│   ├── dataset.py                 # COCO↔YOLO conversion, graph builder
│   ├── augmentation.py            # Rotation, brightness, noise, scale, flip
│   └── constellation_dataset/
│       └── dataset.yaml           # YOLOv8 dataset config
│
├── models/
│   ├── yolo/                      # YOLO weights (constellation_yolo.pt)
│   ├── detr/                      # DETR weights (constellation_detr.pt)
│   └── rcnn/                      # Faster R-CNN weights (constellation_rcnn.pt)
│
├── configs/
│   └── config.yaml                # Master configuration
│
└── tests/
    └── test_pipeline.py           # Unit + integration tests
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `torch-geometric` may need a separate CUDA-specific installation.
> See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Usage

### Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`. The interface provides:

- **Detect tab** — upload a night-sky image, toggle detectors (YOLO / DETR / RCNN), adjust confidence thresholds, and run the full pipeline with a progress indicator.
- **Performance tab** — confidence score charts and per-model detection breakdowns.
- **About tab** — pipeline overview and tips.

Sidebar controls let you enable/disable individual detectors, configure fusion agreement, and tune confidence thresholds per model.

### CLI Inference

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
```

### Dataset Preparation

```bash
# Convert COCO annotations to YOLO format
python main.py convert --direction coco2yolo \
  --coco data/annotations.json --out data/labels/

# Build star graph dataset (for experimental GNN use)
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

### YOLO, DETR & Faster R-CNN
- Bounding box annotations in COCO or YOLO format
- Multiple constellations per image supported
- Built-in augmentations: ±180° rotation, brightness ×0.3–1.5, Gaussian noise, scaling ×0.7–1.3, horizontal/vertical flip

---

## Fusion Rules

A detection is accepted when **any** of the following hold:

| Rule | Condition |
|---|---|
| **Model Agreement** | ≥ 2 detectors produce overlapping boxes (IoU ≥ 0.40) for the same label |
| **High Confidence** | Single detector confidence > 0.90 |

The minimum agreement threshold is configurable in both the config file and the Streamlit sidebar.

---

## Configuration

All hyperparameters are in [configs/config.yaml](configs/config.yaml):

| Section | Key parameters |
|---|---|
| `preprocessing` | `target_size`, `clahe_clip_limit`, `gaussian_blur_kernel` |
| `star_extraction` | `method` (log/dog), `threshold`, `min_star_brightness` |
| `yolo` | `conf_threshold`, `iou_threshold`, `model_weights` |
| `detr` | `conf_threshold`, `num_queries`, `model_weights` |
| `rcnn` | `conf_threshold`, `iou_threshold`, `model_weights` |
| `fusion` | `min_model_agreement`, `iou_merge_threshold` |

---

## Model Weights

| Model | Path | Notes |
|---|---|---|
| YOLO | `models/yolo/constellation_yolo.pt` | Required |
| DETR | `models/detr/constellation_detr.pt` | Falls back to pretrained base if missing |
| Faster R-CNN | `models/rcnn/constellation_rcnn.pt` | Falls back to pretrained base if missing |

---

## Performance Goals

- ✅ Detect multiple constellations per image
- ✅ Supports arbitrary rotation (full ±180° augmentation)
- ✅ Reduces false positives via 3-model ensemble fusion
- ✅ Interactive web interface for rapid experimentation
