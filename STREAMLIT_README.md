# 🌌 Star AI - Streamlit Web Interface

## Quick Start

### 1. Install Streamlit Dependencies
```powershell
pip install streamlit plotly
```

Or update all requirements:
```powershell
pip install -r requirements.txt
```

### 2. Run the Application
```powershell
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## Features

### 📤 Main Interface
- **Upload & Detect Tab**: Upload sky images and run detection
- **Detection Details Tab**: View step-by-step pipeline results
- **Performance Tab**: Analyze confidence scores and GNN validation
- **About Tab**: Learn about the hybrid pipeline, detector scope, and supported constellations

### ⚙️ Sidebar Controls
- **Device Selection**: Choose CPU or CUDA
- **Confidence Thresholds**: Adjust YOLO, DETR, and GNN thresholds
- **Fusion Settings**: Control model agreement requirements

### 🎯 Pipeline Visualization
- Real-time progress tracking through 7 pipeline steps
- Annotated images with bounding boxes
- Color-coded results (green = GNN verified, yellow = not verified)
- Interactive charts and metrics

---

## Usage Guide

1. **Upload Image**
   - Click "Browse files" in the left panel
   - Select a night sky image (JPG, PNG, BMP)
   - Image info will display automatically

2. **Configure Settings** (Optional)
   - Adjust confidence thresholds in sidebar
   - Set minimum model agreement for fusion
   - Choose processing device (CPU/CUDA)

3. **Run Detection**
   - Click "🚀 Start Detection" button
   - Watch real-time progress through 7 steps:
     - Preprocessing
     - Star Extraction
     - YOLO Detection
     - DETR Detection
     - Graph Construction
     - GNN Validation
     - Result Fusion

4. **View Results**
   - See detected constellations with confidence scores
   - Green boxes = GNN verified (high geometric accuracy)
   - Yellow boxes = Detection only (lower geometric confidence)
   - Explore detailed metrics in other tabs

---

## Tips for Best Results

✅ **Do:**
- Use high-resolution images (640x640 or larger)
- Ensure stars are clearly visible
- Use images with minimal light pollution
- Try different confidence thresholds for your images

❌ **Avoid:**
- Very low-resolution images
- Heavily clouded skies
- Extreme light pollution
- Blurry or out-of-focus images

---

## Model Requirements

Ensure trained model weights are in place:
- `models/yolo/constellation_yolo.pt` ✅ (Already in place)
- `models/detr/constellation_detr.pt` (Optional - will use pretrained base if missing)
- `models/gnn/constellation_gnn.pt` (Optional - will skip GNN if missing)

---

## Troubleshooting

### ImportError on `numpy` or `cv2` in Streamlit Cloud

If deployment starts but fails at app import, it usually means the cloud environment
did not install compatible wheels.

Use these project files exactly:
- `requirements.txt` (cloud-safe inference dependencies)
- `runtime.txt` with `python-3.11`
- `packages.txt` for Linux runtime libs

Then fully redeploy (not just refresh):
1. Push latest commit
2. In Streamlit Cloud, verify correct repo + branch + app path
3. Clear cache and reboot app

If logs still show Python 3.14, your app is deploying a different branch/commit.

### Deployment runs but detects nothing

If your trained `.pt` files are too large to commit to GitHub, host them externally
(for example Hugging Face model repo, GitHub Releases assets, S3, or direct file URL)
and provide URLs in Streamlit Cloud secrets.

The app auto-downloads missing weights at startup.

Add this in Streamlit Cloud -> Settings -> Secrets:

```toml
[model_urls]
yolo = "https://your-host/path/constellation_yolo.pt"
detr = "https://your-host/path/constellation_detr.pt"
rcnn = "https://your-host/path/constellation_rcnn.pt"
```

Notes:
- URLs must be direct downloadable links (HTTP/HTTPS).
- Files are downloaded to paths defined in `configs/config.yaml` under each model `model_weights`.
- If a URL is missing, that detector falls back to base/random weights and results may be meaningless.

### App won't start
```powershell
# Reinstall streamlit
pip install --upgrade streamlit plotly
```

### Models fail to load
- Check `configs/config.yaml` device settings match your hardware
- Ensure model paths in config are correct
- Check that `models/yolo/constellation_yolo.pt` exists

### Slow performance
- Use CPU mode in sidebar if CUDA unavailable
- Reduce image resolution before upload
- Close other resource-intensive applications

### No detections found
- Lower confidence thresholds in sidebar
- Try images with clearer, brighter stars
- Check that uploaded image contains constellations

---

## Architecture

```
app.py (Streamlit UI)
    ↓
src/pipeline.py (ConstellationPipeline)
    ↓
┌───────────────────────────────────┐
│  1. Preprocessing                 │
│  2. Star Extraction               │
│  3. YOLO Detection                │
│  4. DETR Detection                │
│  5. Graph Construction            │
│  6. GNN Validation                │
│  7. Result Fusion                 │
└───────────────────────────────────┘
    ↓
PipelineResult (Final detections)
```

---

## Keyboard Shortcuts

- `Ctrl + R` - Refresh/restart app
- `Ctrl + Shift + R` - Hard refresh (clear cache)
- `Ctrl + C` (in terminal) - Stop server

---

## Advanced Configuration

Edit `configs/config.yaml` to adjust:
- Model weights paths
- Confidence thresholds (defaults)
- GNN geometry threshold
- Fusion parameters
- Preprocessing settings
- Graph construction parameters

Changes require app restart to take effect.

---

**Built with:** PyTorch, Ultralytics YOLO, Transformers, PyTorch Geometric, Streamlit, Plotly
