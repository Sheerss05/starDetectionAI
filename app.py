"""
🌌 Star AI - Multi-Constellation Detection System
Main Streamlit Application
"""

import streamlit as st
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageOps
import time
import json
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
from collections.abc import Mapping
import yaml

try:
    import cv2
except Exception:
    cv2 = None

# Import pipeline
from src.pipeline import ConstellationPipeline, PipelineResult

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Star AI - Constellation Recognition",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0F172A; }

.stButton>button {
    width: 100%;
    background-color: #3B82F6;
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
}
.stButton>button:hover { background-color: #2563EB; }

h1 { color: #FBBF24; }
h2, h3 { color: #3B82F6; }

div::-webkit-scrollbar { width: 8px; }
div::-webkit-scrollbar-track { background: #1E293B; border-radius: 10px; }
div::-webkit-scrollbar-thumb { background: #3B82F6; border-radius: 10px; }
div::-webkit-scrollbar-thumb:hover { background: #2563EB; }
</style>
""", unsafe_allow_html=True)

# ── Labels excluded from detection results ──────────────────────────────────
EXCLUDED_LABELS: set = {"pleiades", "moon"}

# ── Constellation metadata ────────────────────────────────────────────────────
CONSTELLATION_INFO = {
    "aquarius":      "Aquarius — the Water-Bearer; one of the 17 deployed classes",
    "aquila":        "Aquila — the Eagle; home of bright star Altair",
    "bootes":        "Boötes — the Herdsman; contains Arcturus",
    "cassiopeia":    "Cassiopeia — the Queen; W-shaped in the north",
    "cygnus":        "Cygnus — the Swan; Northern Cross asterism",
    "gemini":        "Gemini — the Twins; Castor and Pollux stars",
    "leo":           "Leo — the Lion; contains bright star Regulus",
    "lyra":          "Lyra — the Lyre; home of Vega, one of the nearest stars",
    "orion":         "Orion — the Hunter; Betelgeuse and Rigel",
    "pegasus":       "Pegasus — the Winged Horse; autumn sky pattern",
    "perseus":       "Perseus — the Hero; linked to the Perseid meteor shower",
    "sagittarius":   "Sagittarius — the Archer; points to the galactic centre",
    "scorpius":      "Scorpius — the Scorpion; contains red giant Antares",
    "taurus":        "Taurus — the Bull; contains Pleiades and Aldebaran",
    "ursa_major":    "Ursa Major — the Great Bear; Big Dipper asterism",
    "ursa_minor":    "Ursa Minor — the Little Bear; home of Polaris",
    "virgo":         "Virgo — the Maiden; one of the largest zodiac constellations",
}

# ── Session state ─────────────────────────────────────────────────────────────
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'uploaded_image_bgr' not in st.session_state:
    st.session_state.uploaded_image_bgr = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'pipeline_device' not in st.session_state:
    st.session_state.pipeline_device = None

@st.cache_resource
def load_pipeline(config_path: str, device: str = "cpu"):
    try:
        _ensure_model_weights(config_path)
        return ConstellationPipeline(config_path=config_path, device=device)
    except Exception as exc:
        st.error(f"Error loading pipeline: {exc}")
        return None


def _read_config_from_path(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _is_http_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False


def _normalize_model_url(url: str) -> str:
    """Convert known non-direct model links to direct downloadable URLs."""
    if not isinstance(url, str):
        return ""
    u = url.strip()
    if not u:
        return ""

    # Hugging Face web URLs often use /blob/ or /raw/; downloads should use /resolve/.
    if "huggingface.co" in u:
        u = u.replace("/blob/", "/resolve/")
        u = u.replace("/raw/", "/resolve/")

    return u


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(_normalize_model_url(url), dest)


def _get_model_download_url(model_key: str, cfg_section: dict) -> str:
    # Priority:
    # 1) Streamlit secrets [model_urls]
    # 2) Environment variables via st.secrets root entries
    # 3) config.yaml per-model download_url
    model_urls = st.secrets.get("model_urls", {})
    if isinstance(model_urls, Mapping):
        url_from_secrets = model_urls.get(model_key)
    else:
        # Streamlit may expose section objects that behave like attributes.
        url_from_secrets = getattr(model_urls, model_key, None)
    if isinstance(url_from_secrets, str):
        url_from_secrets = _normalize_model_url(url_from_secrets)
        if _is_http_url(url_from_secrets):
            return url_from_secrets

    env_key = f"STARAI_{model_key.upper()}_URL"
    url_from_root_secrets = st.secrets.get(env_key)
    if isinstance(url_from_root_secrets, str):
        url_from_root_secrets = _normalize_model_url(url_from_root_secrets)
        if _is_http_url(url_from_root_secrets):
            return url_from_root_secrets

    cfg_url = cfg_section.get("download_url") if isinstance(cfg_section, Mapping) else None
    if isinstance(cfg_url, str):
        cfg_url = _normalize_model_url(cfg_url)
        if _is_http_url(cfg_url):
            return cfg_url

    return ""


def _ensure_model_weights(config_path: str) -> None:
    cfg = _read_config_from_path(config_path)
    if not cfg:
        return

    specs = [
        ("yolo", cfg.get("yolo", {})),
        ("detr", cfg.get("detr", {})),
        ("rcnn", cfg.get("rcnn", {})),
    ]

    downloaded = []
    missing_without_url = []

    for model_key, section in specs:
        if not isinstance(section, dict):
            continue
        weights_rel = section.get("model_weights")
        if not weights_rel:
            continue

        weights_path = Path(weights_rel)
        if weights_path.exists():
            continue

        url = _get_model_download_url(model_key, section)
        if not url:
            missing_without_url.append(model_key)
            continue

        try:
            _download_file(url, weights_path)
            downloaded.append(f"{model_key.upper()} -> {weights_path}")
        except Exception as exc:
            st.warning(f"Could not download {model_key.upper()} weights: {exc}")

    if missing_without_url:
        msg = (
            "Missing trained model files for: "
            + ", ".join(m.upper() for m in missing_without_url)
            + ". Configure URLs in Streamlit secrets under [model_urls]"
            + " so deployment can download large .pt files."
        )
        st.warning(msg)


def load_config() -> dict:
    cfg_path = Path("configs/config.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


@st.cache_data(show_spinner=False)
def load_dataset_profile(annotations_dir: str = "data/constellation_dataset/annotations") -> dict:
    """Read dataset split stats from COCO annotations."""
    ann_root = Path(annotations_dir)
    split_rows = []
    class_count = 0
    total_images = 0
    total_annotations = 0

    for split in ("train", "val", "test"):
        ann_path = ann_root / f"{split}.json"
        if not ann_path.exists():
            continue

        try:
            with open(ann_path, encoding="utf-8") as f:
                coco = json.load(f)
        except Exception:
            continue

        images = int(len(coco.get("images", [])))
        annotations = int(len(coco.get("annotations", [])))
        categories = int(len(coco.get("categories", [])))

        class_count = max(class_count, categories)
        total_images += images
        total_annotations += annotations

        split_rows.append({
            "Split": split.title(),
            "Images": images,
            "Annotations": annotations,
            "Avg Boxes / Image": round((annotations / images) if images else 0.0, 2),
        })

    return {
        "rows": split_rows,
        "class_count": class_count,
        "total_images": total_images,
        "total_annotations": total_annotations,
    }


@st.cache_data(show_spinner=False)
def load_offline_benchmarks(path: str = "configs/model_benchmarks.yaml") -> tuple[dict, pd.DataFrame]:
    """Load offline benchmark metrics (CSV preferred, YAML fallback)."""
    csv_path = Path("comparison_results.csv")

    def _normalize_model_name(name: str) -> str:
        key = str(name).strip().lower().replace("_", "-")
        if key in {"yolo"}:
            return "YOLO"
        if key in {"detr"}:
            return "DETR"
        if key in {"rcnn", "faster r-cnn", "fasterrcnn", "faster-rcnn"}:
            return "RCNN"
        return str(name).strip()

    # Prefer benchmark output generated by evaluate_models.py
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            models = []
            for _, row in df.iterrows():
                model_name = _normalize_model_name(row.get("Model", "Unknown"))
                models.append({
                    "model": model_name,
                    "map50": row.get("mAP@0.5"),
                    "precision": row.get("Precision"),
                    "recall": row.get("Recall"),
                    "f1": row.get("F1-Score"),
                    "ap_small": row.get("AP-Small"),
                    "latency_ms": row.get("Latency (ms/img)"),
                    "params_m": row.get("Params (M)"),
                })

            rows = []
            for item in models:
                rows.append({
                    "Model": item.get("model", "Unknown"),
                    "mAP@50": item.get("map50", "-"),
                    "Precision": item.get("precision", "-"),
                    "Recall": item.get("recall", "-"),
                    "F1": item.get("f1", "-"),
                    "Latency (ms/img)": item.get("latency_ms", "-"),
                })

            return {"source": str(csv_path), "models": models}, pd.DataFrame(rows)
        except Exception:
            # Fallback to YAML if CSV parsing fails.
            pass

    p = Path(path)
    if not p.exists():
        return {}, pd.DataFrame()

    try:
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}, pd.DataFrame()

    rows = []
    for item in data.get("models", []) or []:
        if not isinstance(item, dict):
            continue
        rows.append({
            "Model": item.get("model", "Unknown"),
            "mAP@50": item.get("map50", "-"),
            "Precision": item.get("precision", "-"),
            "Recall": item.get("recall", "-"),
            "F1": item.get("f1", "-"),
            "Latency (ms/img)": item.get("latency_ms", "-"),
        })

    return data, pd.DataFrame(rows)


def _bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_detection_comparison_table(result: PipelineResult | None) -> pd.DataFrame:
    """Build model comparison with true benchmark metrics + runtime diagnostics."""
    if result is None:
        return pd.DataFrame()

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return None

    rows = []
    
    # First try to load from comparison_results.csv (generated by evaluate_models.py)
    csv_path = Path("comparison_results.csv")
    if csv_path.exists():
        try:
            eval_df = pd.read_csv(csv_path)
            model_map = {
                "YOLO": "YOLO",
                "Faster R-CNN": "RCNN",
                "DETR": "DETR"
            }
            
            for idx, row in eval_df.iterrows():
                model_name_csv = str(row.get("Model", ""))
                model_key = model_map.get(model_name_csv, model_name_csv)
                
                map50 = _to_float(row.get("mAP@0.5"))
                f1 = _to_float(row.get("F1-Score"))
                ap_small = _to_float(row.get("AP-Small"))
                conf_benchmark = _to_float(row.get("Avg Confidence (TP)"))
                latency_benchmark = _to_float(row.get("Latency (ms/img)"))
                params_m = _to_float(row.get("Params (M)"))
                
                # Use benchmark values, optionally override with runtime if available
                lat_s = result.model_times.get(model_key.lower(), 0.0) if isinstance(result.model_times, dict) else 0.0
                runtime_latency_ms = lat_s * 1000.0 if lat_s and lat_s > 0 else None
                latency_ms = runtime_latency_ms if runtime_latency_ms is not None else latency_benchmark
                
                # Confidence from current run (if available), else benchmark
                dets_map = {"YOLO": result.yolo_raw, "RCNN": result.rcnn_raw, "DETR": result.detr_raw}
                preds = list(dets_map.get(model_key, []))
                conf_avg = float(np.mean([d.confidence for d in preds])) if preds else conf_benchmark
                
                # Calculate overall performance score
                score_parts = []
                for val, weight in ((map50, 0.40), (f1, 0.40), (ap_small, 0.20)):
                    if val is not None:
                        score_parts.append((float(val), weight))
                
                latency_bonus = 0.0
                if isinstance(latency_ms, (int, float)) and latency_ms > 0:
                    latency_bonus = min(0.10, 12.0 / latency_ms * 0.05)
                
                if score_parts:
                    total_w = sum(w for _, w in score_parts)
                    perf = (sum(v * w for v, w in score_parts) / total_w) + latency_bonus
                    perf = max(0.0, min(1.0, perf))
                    if perf >= 0.75:
                        overall = f"High ({perf:.2f})"
                    elif perf >= 0.55:
                        overall = f"Medium ({perf:.2f})"
                    else:
                        overall = f"Low ({perf:.2f})"
                else:
                    overall = "-"
                
                rows.append({
                    "Model": model_name_csv,
                    "mAP @0.5": f"{map50:.4f}" if map50 is not None else "-",
                    "Precision": f"{_to_float(row.get('Precision')):.4f}" if _to_float(row.get('Precision')) else "-",
                    "Recall": f"{_to_float(row.get('Recall')):.4f}" if _to_float(row.get('Recall')) else "-",
                    "F1-Score": f"{f1:.4f}" if f1 is not None else "-",
                    "AP-Small": f"{ap_small:.4f}" if ap_small is not None else "-",
                    "Confidence": f"{conf_avg:.4f}",
                    "Latency(ms)": f"{latency_ms:.1f}" if latency_ms is not None else "-",
                    "Params(M)": f"{params_m:.2f}" if params_m is not None else "-",
                    "Overall Performance": overall,
                })
            
            return pd.DataFrame(rows)
        except Exception as e:
            st.warning(f"Could not load comparison_results.csv: {e}")
    
    # Fallback to configs/model_benchmarks.yaml if CSV not available
    bench_meta, _ = load_offline_benchmarks("configs/model_benchmarks.yaml")
    bench_rows = bench_meta.get("models", []) if isinstance(bench_meta, dict) else []
    bench_lookup = {}
    for row in bench_rows:
        if isinstance(row, dict) and isinstance(row.get("model"), str):
            bench_lookup[row["model"].strip().lower()] = row

    for model_name, dets in (("YOLO", result.yolo_raw), ("Faster R-CNN", result.rcnn_raw), ("DETR", result.detr_raw)):
        preds = list(dets)
        b = bench_lookup.get(model_name.lower(), {})

        map50 = _to_float(b.get("map50") if isinstance(b, dict) else None)
        f1 = _to_float(b.get("f1") if isinstance(b, dict) else None)
        ap_small = _to_float(b.get("ap_small") if isinstance(b, dict) else None)
        conf_avg = float(np.mean([d.confidence for d in preds])) if preds else 0.0

        lat_s = result.model_times.get(model_name.lower(), 0.0) if isinstance(result.model_times, dict) else 0.0
        runtime_latency_ms = lat_s * 1000.0 if lat_s and lat_s > 0 else None
        benchmark_latency_ms = _to_float(b.get("latency_ms") if isinstance(b, dict) else None)
        latency_ms = runtime_latency_ms if runtime_latency_ms is not None else benchmark_latency_ms
        params_m = _to_float(b.get("params_m") if isinstance(b, dict) else None)

        required_vals = [map50, f1, ap_small, conf_avg, latency_ms, params_m]
        if any(v is None for v in required_vals):
            continue

        score_parts = []
        for val, weight in ((map50, 0.40), (f1, 0.40), (ap_small, 0.20)):
            score_parts.append((float(val), weight))

        latency_bonus = 0.0
        if isinstance(latency_ms, (int, float)) and latency_ms > 0:
            latency_bonus = min(0.10, 12.0 / latency_ms * 0.05)

        if score_parts:
            total_w = sum(w for _, w in score_parts)
            perf = (sum(v * w for v, w in score_parts) / total_w) + latency_bonus
            perf = max(0.0, min(1.0, perf))
            if perf >= 0.75:
                overall = f"High ({perf:.2f})"
            elif perf >= 0.55:
                overall = f"Medium ({perf:.2f})"
            else:
                overall = f"Low ({perf:.2f})"
        else:
            overall = "-"

        rows.append({
            "Model": model_name,
            "mAP @0.5": f"{map50:.3f}",
            "F1-Score": f"{f1:.3f}",
            "AP-Small": f"{ap_small:.3f}",
            "Confidence": f"{conf_avg:.3f}",
            "Latency(ms)": f"{latency_ms:.3f}",
            "Params(M)": f"{params_m:.3f}",
            "Overall Performance": overall,
        })

    return pd.DataFrame(rows)


def build_missing_real_values_report(result: PipelineResult | None) -> pd.DataFrame:
    """Report which required real-value fields are still missing per model."""
    if result is None:
        return pd.DataFrame()

    bench_meta, _ = load_offline_benchmarks("configs/model_benchmarks.yaml")
    bench_rows = bench_meta.get("models", []) if isinstance(bench_meta, dict) else []
    bench_lookup = {}
    for row in bench_rows:
        if isinstance(row, dict) and isinstance(row.get("model"), str):
            bench_lookup[row["model"].strip().lower()] = row

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return None

    report_rows = []
    for model_name, dets in (("YOLO", result.yolo_raw), ("DETR", result.detr_raw), ("RCNN", result.rcnn_raw)):
        b = bench_lookup.get(model_name.lower(), {})

        map50 = _to_float(b.get("map50") if isinstance(b, dict) else None)
        f1 = _to_float(b.get("f1") if isinstance(b, dict) else None)
        ap_small = _to_float(b.get("ap_small") if isinstance(b, dict) else None)
        params_m = _to_float(b.get("params_m") if isinstance(b, dict) else None)

        conf_avg = float(np.mean([d.confidence for d in dets])) if dets else 0.0
        lat_s = result.model_times.get(model_name.lower(), 0.0) if isinstance(result.model_times, dict) else 0.0
        runtime_latency_ms = lat_s * 1000.0 if lat_s and lat_s > 0 else None
        bench_latency_ms = _to_float(b.get("latency_ms") if isinstance(b, dict) else None)
        latency_ms = runtime_latency_ms if runtime_latency_ms is not None else bench_latency_ms

        missing = []
        if map50 is None:
            missing.append("map50")
        if f1 is None:
            missing.append("f1")
        if ap_small is None:
            missing.append("ap_small")
        if params_m is None:
            missing.append("params_m")
        if conf_avg is None:
            missing.append("confidence(runtime)")
        if latency_ms is None:
            missing.append("latency(runtime or benchmark)")

        if missing:
            report_rows.append({
                "Model": model_name,
                "Missing Fields": ", ".join(missing),
            })

    return pd.DataFrame(report_rows)


def build_runtime_comparison(result: PipelineResult | None) -> pd.DataFrame:
    """Build per-model runtime comparison from the latest detection run."""
    if result is None:
        return pd.DataFrame()

    fused_labels = {d.label for d in result.detections}
    model_dets = {
        "YOLO": result.yolo_raw,
        "DETR": result.detr_raw,
        "RCNN": result.rcnn_raw,
    }

    rows = []
    for model_name, dets in model_dets.items():
        confs = [d.confidence for d in dets]
        labels = {d.label for d in dets}
        overlap = len(labels & fused_labels)
        support = (overlap / len(fused_labels) * 100.0) if fused_labels else 0.0
        rows.append({
            "Model": model_name,
            "Raw Detections": len(dets),
            "Avg Confidence": round(float(np.mean(confs)) if confs else 0.0, 3),
            "Max Confidence": round(float(np.max(confs)) if confs else 0.0, 3),
            "Labels Matching Fused": f"{overlap}/{len(fused_labels)}",
            "Fused Label Support (%)": round(support, 1),
        })

    return pd.DataFrame(rows)

# ── Image enhancement helper ─────────────────────────────────────────────────
def apply_enhancements(
    img_rgb: np.ndarray,
    use_blur: bool, blur_kernel: int,
    use_clahe: bool, clahe_clip: float, clahe_tile: int,
    use_gamma: bool, gamma: float,
    use_sharpen: bool, sharpen_strength: float,
) -> np.ndarray:
    """Apply Gaussian blur → CLAHE → Gamma → Sharpening to an RGB uint8 image."""
    img = img_rgb.copy()

    if cv2 is None:
        # Fallback path for environments without OpenCV runtime libs.
        pil_img = Image.fromarray(img.astype(np.uint8))

        if use_blur and blur_kernel > 1:
            radius = max(0.1, blur_kernel / 6.0)
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

        if use_clahe:
            # Approximate CLAHE safely without extra dependencies.
            r, g, b = pil_img.split()
            r = ImageOps.equalize(r)
            g = ImageOps.equalize(g)
            b = ImageOps.equalize(b)
            pil_img = Image.merge("RGB", (r, g, b))

        if use_gamma and abs(gamma - 1.0) > 1e-3:
            inv_gamma = 1.0 / gamma
            table = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
            )
            img = np.asarray(pil_img)
            img = table[img]
            pil_img = Image.fromarray(img)

        if use_sharpen and sharpen_strength > 0:
            sharpness = 1.0 + float(sharpen_strength)
            pil_img = pil_img.filter(
                ImageFilter.UnsharpMask(radius=1, percent=int(150 * sharpness), threshold=3)
            )

        return np.asarray(pil_img, dtype=np.uint8)

    # 1. Gaussian Blur — remove noise before contrast boosting
    if use_blur and blur_kernel > 1:
        img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), sigmaX=0)

    # 2. CLAHE — boost local contrast on the luminance channel
    if use_clahe:
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch_lab = cv2.split(img_lab)
        _clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile)
        )
        l_ch = _clahe.apply(l_ch)
        img = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch_lab]), cv2.COLOR_LAB2RGB)

    # 3. Gamma Correction — brighten dim stars
    if use_gamma and abs(gamma - 1.0) > 1e-3:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
        )
        img = cv2.LUT(img, table)

    # 4. Sharpening — unsharp mask to make star edges more distinct
    if use_sharpen and sharpen_strength > 0:
        blurred_s = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = cv2.addWeighted(
            img, 1.0 + sharpen_strength, blurred_s, -sharpen_strength, 0
        )
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img

# ── Drawing helpers ───────────────────────────────────────────────────────────
PALETTE = [
    (52, 211, 153),   # emerald
    (251, 191, 36),   # amber
    (96, 165, 250),   # blue
    (248, 113, 113),  # rose
    (167, 139, 250),  # violet
    (45, 212, 191),   # teal
]


def draw_boxes(base_img: np.ndarray, detections, palette=PALETTE) -> np.ndarray:
    if cv2 is None:
        pil = Image.fromarray(base_img.astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det.bbox)
            color = palette[idx % len(palette)]
            txt = f"{det.label.replace('_', ' ').title()}  {det.confidence:.0%}"
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            draw.rectangle([(x1, max(0, y1 - 18)), (x1 + 220, y1)], fill=color)
            draw.text((x1 + 4, max(0, y1 - 16)), txt, fill=(15, 23, 42))
        return np.array(pil)

    img = base_img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det.bbox)
        color = palette[idx % len(palette)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        txt = f"{det.label.replace('_', ' ').title()}  {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(txt, font, 0.55, 2)
        py1 = max(y1 - th - 10, 0)
        cv2.rectangle(img, (x1, py1), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, txt, (x1 + 4, y1 - 4), font, 0.55, (15, 23, 42), 2, cv2.LINE_AA)
    return img


def det_cards(detections, palette=PALETTE):
    """Render styled detection cards (right-side panel)."""
    if not detections:
        st.markdown(
            "<div style='background:#1E293B;border-left:4px solid #475569;"
            "border-radius:10px;padding:14px 18px;color:#94A3B8;'>"
            "⚪ No detections from this model.</div>",
            unsafe_allow_html=True,
        )
        return
    for i, det in enumerate(detections):
        pct = det.confidence * 100
        bar_color = "#34D399" if pct >= 70 else "#FBBF24" if pct >= 40 else "#F87171"
        icon = "🟢" if pct >= 70 else "🟡" if pct >= 40 else "🔴"
        r, g, b_ch = palette[i % len(palette)]
        hex_color = f"#{r:02x}{g:02x}{b_ch:02x}"
        b = det.bbox
        w_px, h_px = b[2] - b[0], b[3] - b[1]
        desc = CONSTELLATION_INFO.get(det.label, "")
        st.markdown(
            f"""<div style='
                background:#1E293B;
                border-left:4px solid {hex_color};
                border-radius:10px;
                padding:14px 18px;
                margin-bottom:10px;
                display:flex;
                align-items:center;
                gap:16px;
            '>
              <div style='font-size:1.5rem;line-height:1'>{icon}</div>
              <div style='flex:1;min-width:0;'>
                <div style='font-size:1rem;font-weight:700;color:#F1F5F9;margin-bottom:2px;'>
                  {det.label.replace("_", " ").title()}
                </div>
                {"<div style='font-size:0.75rem;color:#64748B;margin-bottom:6px;'>" + desc + "</div>" if desc else ""}
                <div style='background:#0F172A;border-radius:6px;height:10px;margin-bottom:4px;'>
                  <div style='background:{bar_color};width:{pct:.0f}%;height:10px;border-radius:6px;'></div>
                </div>
                <span style='font-size:0.78rem;color:#94A3B8;'>
                  Confidence: {det.confidence:.4f} ({pct:.1f}%)
                </span>
              </div>
              <div style='text-align:right;font-size:0.78rem;color:#64748B;white-space:nowrap;flex-shrink:0;'>
                <div>x1={b[0]:.0f} &nbsp;y1={b[1]:.0f}</div>
                <div>x2={b[2]:.0f} &nbsp;y2={b[3]:.0f}</div>
                <div style='color:#94A3B8;margin-top:2px;'>{w_px:.0f} × {h_px:.0f} px</div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


def model_section(label: str, icon: str, detections, base_rgb, expanded: bool = True):
    """Expandable model result section — left: annotated image, right: detection cards."""
    count = len(detections)
    with st.expander(f"{icon} {label} — {count} detection(s)", expanded=expanded):
        c_img, c_info = st.columns([1, 1], gap="large")
        with c_img:
            if detections and base_rgb is not None:
                annotated = draw_boxes(base_rgb, detections)
                st.image(annotated, caption=f"{label} — {count} detection(s)",
                         use_container_width=True)
            elif base_rgb is not None:
                st.image(base_rgb, caption="No detections", use_container_width=True)
            else:
                st.info("No image available.")
        with c_info:
            st.markdown(f"**{count} constellation(s) detected**")
            det_cards(detections)

def main():
    config = load_config()
    constellations = config.get("constellations", list(CONSTELLATION_INFO.keys()))

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;'>🌌 Star AI — Multi-Constellation Detection</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#94A3B8;'>"
        "Hybrid YOLO + DETR + RCNN Pipeline for Constellation Recognition"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Settings")

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        st.subheader("Model Selection")
        st.caption("Choose which detectors to run:")
        if cv2 is None:
            st.warning("OpenCV not available in runtime. Using fallback image pipeline.")

        use_yolo = st.checkbox("🎯 YOLO Detector", value=True)
        if use_yolo:
            yolo_conf = st.slider("YOLO Confidence Threshold", 0.0, 1.0,
                                   config.get("yolo", {}).get("conf_threshold", 0.30),
                                   step=0.05, key="yolo_conf")

        use_detr = st.checkbox("🔍 DETR Detector", value=True)
        if use_detr:
            detr_conf = st.slider("DETR Confidence Threshold", 0.0, 1.0,
                                   config.get("detr", {}).get("conf_threshold", 0.30),
                                   step=0.05, key="detr_conf")

        use_rcnn = st.checkbox("🔬 RCNN Detector", value=True)
        if use_rcnn:
            rcnn_conf = st.slider("RCNN Confidence Threshold", 0.0, 1.0,
                                   config.get("rcnn", {}).get("conf_threshold", 0.30),
                                   step=0.05, key="rcnn_conf")

        if not (use_yolo or use_detr or use_rcnn):
            st.warning("⚠️ Enable at least one detector.")

        st.markdown("---")
        st.subheader("🌟 Image Enhancement")

        use_sharpen = st.checkbox(
            "🔪 Sharpening", value=False,
            help="Unsharp-mask to make star edges more distinct for detection.",
        )
        if use_sharpen:
            sharpen_strength = st.slider(
                "Sharpen Strength", 0.1, 2.0, 0.5, step=0.1, key="sharpen_str",
                help="Higher = stronger edge enhancement.",
            )
        else:
            sharpen_strength = 0.0

        use_gamma = st.checkbox(
            "☀️ Gamma Correction", value=False,
            help="Brighten dim stars by adjusting the gamma curve.",
        )
        if use_gamma:
            gamma = st.slider(
                "Gamma", 0.5, 3.0, 1.0, step=0.1, key="gamma",
                help="< 1.0 darkens, > 1.0 brightens. 1.0 = no change.",
            )
        else:
            gamma = 1.0

        use_clahe = st.checkbox(
            "✨ CLAHE Contrast Enhancement", value=False,
            help="Adaptive histogram equalisation to boost faint-star visibility.",
        )
        if use_clahe:
            clahe_clip = st.slider(
                "CLAHE Clip Limit", 1.0, 8.0,
                float(config.get("preprocessing", {}).get("clahe_clip_limit", 2.0)),
                step=0.5, key="clahe_clip",
                help="Higher values = more aggressive contrast boost.",
            )
            clahe_tile = st.slider(
                "CLAHE Tile Size", 4, 16,
                int(config.get("preprocessing", {}).get("clahe_tile_grid", [8, 8])[0]),
                step=2, key="clahe_tile",
                help="Context tile size for local histogram equalisation.",
            )
        else:
            clahe_clip = 0.0
            clahe_tile = 8

        use_blur = st.checkbox(
            "🌫️ Gaussian Blur (Noise Reduction)", value=False,
            help="Smooth pixel-level noise before detection.",
        )
        if use_blur:
            blur_kernel = st.select_slider(
                "Blur Kernel Size", options=[3, 5, 7, 9], value=3, key="blur_kernel",
                help="Larger kernel = stronger blurring.",
            )
        else:
            blur_kernel = 1

        st.markdown("---")
        st.subheader("Fusion Settings")
        use_fusion = st.checkbox("🔗 Enable Fusion", value=True,
                                  help="Combine results from all active detectors. Disable to return raw detections from each model independently.")
        if use_fusion:
            min_agreement = st.slider(
                "Min Model Agreement", 1, 3,
                config.get("fusion", {}).get("min_model_agreement", 2),
            )
        else:
            min_agreement = 1


    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔭 Detect", "📊 Performance", "ℹ️ About"])

    # ══ TAB 1: Upload + Detect ═════════════════════════════════════════════════
    with tab1:

        # ── SECTION 1: Upload box (full width) ───────────────────────────────
        st.subheader("📤 Upload Night Sky Image")
        col_up, col_prev = st.columns([1, 1], gap="large")

        with col_up:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["png", "jpg", "jpeg", "bmp"],
                help="Upload a clear image of the night sky",
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = np.array(image)
                if st.session_state.uploaded_image.ndim == 3:
                    if cv2 is not None:
                        st.session_state.uploaded_image_bgr = cv2.cvtColor(
                            st.session_state.uploaded_image, cv2.COLOR_RGB2BGR
                        )
                    else:
                        st.session_state.uploaded_image_bgr = st.session_state.uploaded_image[:, :, ::-1]
                else:
                    st.session_state.uploaded_image_bgr = st.session_state.uploaded_image

            if st.session_state.uploaded_image is not None:
                if st.button("🚀 Start Detection", type="primary"):
                    with st.spinner("🔭 Analysing constellation patterns…"):
                        if st.session_state.pipeline is None or st.session_state.pipeline_device != device:
                            _ph = st.empty()
                            _ph.text("Loading models…")
                            st.session_state.pipeline = load_pipeline(
                                "configs/config.yaml", device=device
                            )
                            st.session_state.pipeline_device = device
                            _ph.empty()

                        if st.session_state.pipeline is None:
                            st.error("❌ Failed to load pipeline. Check logs.")
                        else:
                            if use_yolo and getattr(st.session_state.pipeline, "yolo", None) is not None:
                                if getattr(st.session_state.pipeline.yolo, "model", None) is None:
                                    st.warning(
                                        "YOLO fine-tuned weights are missing on this deployment. "
                                        "YOLO will be skipped until the trained checkpoint is provided."
                                    )
                                    use_yolo = False

                            if use_fusion:
                                st.session_state.pipeline.fusion.min_agreement = min_agreement
                            # Apply CLAHE + blur settings to preprocessor
                            st.session_state.pipeline.preprocessor.gaussian_kernel = blur_kernel
                            if cv2 is not None:
                                st.session_state.pipeline.preprocessor.clahe = cv2.createCLAHE(
                                    clipLimit=clahe_clip if use_clahe else 0.0,
                                    tileGridSize=(clahe_tile, clahe_tile),
                                )
                            else:
                                st.session_state.pipeline.preprocessor.clahe = None
                                st.session_state.pipeline.preprocessor.clahe_clip = clahe_clip if use_clahe else 0.0
                                st.session_state.pipeline.preprocessor.clahe_tile = (clahe_tile, clahe_tile)
                            if use_yolo:
                                st.session_state.pipeline.yolo.conf_threshold = yolo_conf
                            if use_detr:
                                st.session_state.pipeline.detr.conf_threshold = detr_conf
                            if use_rcnn:
                                st.session_state.pipeline.rcnn.conf_threshold = rcnn_conf

                            progress_bar = st.progress(0)
                            status_text  = st.empty()

                            status_text.text("🔬 Step 1/6: Preprocessing…")
                            progress_bar.progress(0.17)
                            time.sleep(0.2)

                            status_text.text("⭐ Step 2/6: Extracting stars…")
                            progress_bar.progress(0.33)
                            time.sleep(0.2)

                            status_text.text("🎯 Step 3/6: YOLO detection…")
                            progress_bar.progress(0.50)

                            # Apply gamma + sharpening to the input image
                            # (CLAHE + blur are handled inside the preprocessor)
                            _input_rgb = apply_enhancements(
                                st.session_state.uploaded_image,
                                use_blur=False, blur_kernel=1,
                                use_clahe=False, clahe_clip=0.0, clahe_tile=8,
                                use_gamma=use_gamma, gamma=gamma,
                                use_sharpen=use_sharpen, sharpen_strength=sharpen_strength,
                            )
                            _input_bgr = cv2.cvtColor(_input_rgb, cv2.COLOR_RGB2BGR) if cv2 is not None else _input_rgb
                            result = st.session_state.pipeline.run(
                                _input_bgr,
                                use_yolo=use_yolo,
                                use_detr=use_detr,
                                use_rcnn=use_rcnn,
                                use_fusion=use_fusion,
                            )

                            status_text.text("🔍 Step 4/6: DETR detection…")
                            progress_bar.progress(0.67)
                            time.sleep(0.2)

                            status_text.text("🔬 Step 5/6: RCNN detection…")
                            progress_bar.progress(0.83)
                            time.sleep(0.2)

                            status_text.text("🔗 Step 6/6: Fusing results…")
                            progress_bar.progress(1.0)
                            time.sleep(0.2)

                            # Strip excluded labels from all result lists
                            def _filter(lst): return [d for d in lst if d.label not in EXCLUDED_LABELS]
                            result.detections = _filter(result.detections)
                            result.yolo_raw   = _filter(result.yolo_raw)
                            result.detr_raw   = _filter(result.detr_raw)
                            result.rcnn_raw   = _filter(result.rcnn_raw)

                            st.session_state.result = result
                            progress_bar.empty()
                            status_text.empty()
                            st.success("✅ Detection Complete!")
            else:
                st.info("👈 Upload an image to begin.")

        with col_prev:
            if st.session_state.uploaded_image is not None:
                img_preview = apply_enhancements(
                    st.session_state.uploaded_image,
                    use_blur=use_blur, blur_kernel=blur_kernel,
                    use_clahe=use_clahe, clahe_clip=clahe_clip, clahe_tile=clahe_tile,
                    use_gamma=use_gamma, gamma=gamma,
                    use_sharpen=use_sharpen, sharpen_strength=sharpen_strength,
                )
                active = []
                if use_blur:    active.append(f"Blur k={blur_kernel}")
                if use_clahe:   active.append(f"CLAHE clip={clahe_clip} tile={clahe_tile}")
                if use_gamma:   active.append(f"Gamma={gamma:.1f}")
                if use_sharpen: active.append(f"Sharpen={sharpen_strength:.1f}")
                caption = ("Preview — " + ", ".join(active)) if active else "Preview"
                st.image(Image.fromarray(img_preview), caption=caption, use_container_width=True)
                orig = Image.fromarray(st.session_state.uploaded_image)
                st.info(f"📐 {orig.size[0]} × {orig.size[1]} px")

        # ── Summary metrics ───────────────────────────────────────────────────
        if st.session_state.result:
            result: PipelineResult = st.session_state.result
            st.markdown("---")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: st.metric("⭐ Stars",   len(result.stars))
            with m2: st.metric("🎯 YOLO",    len(result.yolo_raw))
            with m3: st.metric("🔍 DETR",    len(result.detr_raw))
            with m4: st.metric("🔬 RCNN",    len(result.rcnn_raw))
            with m5: st.metric("✅ Final",   len(result.detections))

        st.markdown("---")

        # ── SECTION 2: Per-model expandable panels ────────────────────────────
        if st.session_state.result:
            result: PipelineResult = st.session_state.result
            base_rgb = result.original_rgb

            if use_yolo:
                model_section("YOLO Results",  "🎯", result.yolo_raw, base_rgb, expanded=True)
            if use_detr:
                model_section("DETR Results",  "🔍", result.detr_raw, base_rgb, expanded=True)
            if use_rcnn:
                model_section("RCNN Results",  "🔬", result.rcnn_raw, base_rgb, expanded=True)

            # ── Fused final panel (only shown when fusion is enabled) ────────
            if use_fusion:
                st.markdown("---")
                st.subheader(f"🔗 Fused Final Results — {len(result.detections)} constellation(s) confirmed")
                c_img, c_info = st.columns([1, 1], gap="large")
                with c_img:
                    if result.detections and base_rgb is not None:
                        fused_img = draw_boxes(base_rgb, result.detections)
                        st.image(fused_img, caption="Fused — consensus across all models",
                                 use_container_width=True)
                    elif base_rgb is not None:
                        st.image(base_rgb, caption="No detections", use_container_width=True)
                with c_info:
                    det_cards(result.detections)
        elif st.session_state.uploaded_image is not None:
            st.info("Run detection to see model results below.")

    # ══ TAB 2: Performance ════════════════════════════════════════════════════
    with tab2:
        st.markdown(
            "<h2 style='color:#000000;margin-bottom:2px;'>📊 Performance Dashboard</h2>"
            "<p style='color:#64748B;margin-top:0;margin-bottom:0;'>"
            "Metrics and diagnostics from the last detection run</p>",
            unsafe_allow_html=True,
        )

        if st.session_state.result:
            result: PipelineResult = st.session_state.result

            # ── Data Source Validation ────────────────────────────────────────────
            with st.expander("🔍 Data Source & Validation Info", expanded=False):
                col_src1, col_src2, col_src3 = st.columns(3)
                
                with col_src1:
                    st.markdown("**Runtime Detection Data**")
                    st.caption(f"✓ YOLO raw: {len(result.yolo_raw)} detections")
                    st.caption(f"✓ DETR raw: {len(result.detr_raw)} detections")
                    st.caption(f"✓ RCNN raw: {len(result.rcnn_raw)} detections")
                    st.caption(f"✓ Fused: {len(result.detections)} final detections")
                
                with col_src2:
                    st.markdown("**Benchmark Data Source**")
                    csv_exists = Path("comparison_results.csv").exists()
                    yaml_exists = Path("configs/model_benchmarks.yaml").exists()
                    st.caption(f"{'✓' if csv_exists else '✗'} comparison_results.csv")
                    st.caption(f"{'✓' if yaml_exists else '✗'} model_benchmarks.yaml")
                    if csv_exists:
                        st.caption("📊 Using CSV for all metrics")
                    else:
                        st.caption("📋 Using YAML fallback")
                
                with col_src3:
                    st.markdown("**All Visualizations Use**")
                    st.caption("✓ Runtime data (detections, latency)")
                    st.caption("✓ Validated benchmarks (mAP, P, R, F1)")
                    st.caption("✓ Synchronized class mapping")
                    st.caption("✓ Identical evaluation protocol")

            st.markdown("---")

            # ── KPI row ──────────────────────────────────────────────────────────
            all_confs  = [d.confidence for d in result.detections]
            avg_conf   = float(np.mean(all_confs)) if all_confs else 0.0
            raw_counts = {
                "YOLO": len(result.yolo_raw),
                "DETR": len(result.detr_raw),
                "RCNN": len(result.rcnn_raw),
            }
            best_model = max(raw_counts, key=raw_counts.get)
            total_raw  = sum(raw_counts.values())

            k1, k2, k3, k4, k5 = st.columns(5)
            with k1: st.metric("⏱ Total Time",       f"{result.elapsed_seconds:.2f}s")
            with k2: st.metric("✅ Fused Detections", len(result.detections))
            with k3: st.metric("📊 Avg Confidence",   f"{avg_conf:.1%}")
            with k4: st.metric("⭐ Stars Extracted",  len(result.stars))
            with k5: st.metric("🏆 Top Model",        best_model, delta=f"{raw_counts[best_model]} dets")

            st.markdown("---")

            st.subheader("Model Comparison Table")
            cmp_df = build_detection_comparison_table(result)
            if cmp_df.empty:
                st.info("No complete real-value rows yet. Detection already provides Confidence and Latency. Run evaluate_models.py to generate comparison_results.csv (auto-loaded), or fill configs/model_benchmarks.yaml.")
                missing_df = build_missing_real_values_report(result)
                if not missing_df.empty:
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            st.caption("✓ Metrics from comparison_results.csv (generated by evaluate_models.py). Includes mAP@0.5, Precision, Recall, F1-Score, AP-Small, and model parameters. Confidence reflects current detection run.")

            st.markdown("---")

            # ── RADAR CHART: Model Performance Comparison ─────────────────────────
            if not cmp_df.empty:
                st.subheader("⭐ Model Performance Radar Chart")
                
                # Extract metrics for radar visualization
                radar_data = []
                radar_colors = {"YOLO": "#60A5FA", "Faster R-CNN": "#F472B6", "DETR": "#34D399"}
                
                for idx, row in cmp_df.iterrows():
                    model = row.get("Model", "Unknown")
                    
                    # Parse numeric metrics (they may be formatted as strings)
                    def parse_metric(val):
                        if isinstance(val, str):
                            try:
                                return float(val)
                            except:
                                return 0.0
                        return float(val) if val else 0.0
                    
                    map50 = parse_metric(row.get("mAP @0.5", 0))
                    precision = parse_metric(row.get("Precision", 0))
                    recall = parse_metric(row.get("Recall", 0))
                    f1 = parse_metric(row.get("F1-Score", 0))
                    latency_ms = parse_metric(row.get("Latency(ms)", 1.0))
                    
                    # Speed metric (inverse of latency, normalized: faster = higher score)
                    # Normalize to 0-1 range based on reasonable latency bounds (0-3000 ms)
                    speed = max(0.1, 1.0 - min(latency_ms / 3000.0, 1.0))
                    
                    radar_data.append({
                        "Model": model,
                        "Color": radar_colors.get(model, "#999999"),
                        "mAP@0.5": map50,
                        "Precision": precision,
                        "Recall": recall,
                        "F1-Score": f1,
                        "Speed": speed,
                    })
                
                if radar_data:
                    fig_radar = go.Figure()
                    
                    metrics = ["mAP@0.5", "Precision", "Recall", "F1-Score", "Speed"]
                    
                    for data in radar_data:
                        values = [data[m] for m in metrics]
                        # Close the loop for radar chart
                        values += values[:1]
                        metrics_closed = metrics + [metrics[0]]
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=metrics_closed,
                            name=data["Model"],
                            fill="toself",
                            fillcolor=data["Color"],
                            marker=dict(
                                size=8,
                                color=data["Color"],
                                line=dict(width=2, color="white"),
                            ),
                            opacity=0.5,
                            line_color=data["Color"],
                            line_width=2.5,
                            hoverinfo="all",
                            hovertemplate="<b>%{fullData.name}</b><br>%{theta}<br>Score: %{r:.4f}<extra></extra>",
                            hoveron="points+fills",
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1],
                                tickfont=dict(size=9, color="#94A3B8", family="Arial"),
                                gridcolor="#1E293B",
                                gridwidth=1,
                            ),
                            angularaxis=dict(
                                tickfont=dict(size=10, color="#94A3B8", family="Arial"),
                                gridcolor="#1E293B",
                                gridwidth=1,
                            ),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=480,
                        font=dict(size=11, family="Arial", color="#E2E8F0"),
                        legend=dict(
                            orientation="v",
                            x=1.0,
                            y=1,
                            bgcolor="rgba(0,0,0,0)",
                            bordercolor="rgba(0,0,0,0)",
                            font=dict(size=10, family="Arial"),
                        ),
                        margin=dict(t=20, b=20, l=20, r=140),
                        hovermode="closest",
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                    st.caption(
                        "mAP = detection accuracy | Precision = positives correctness | Recall = ground truth coverage | F1 = harmonic mean | Speed = inference speed"
                    )

            st.markdown("---")

            # ── DETAILED METRICS COMPARISON: Multi-metric grouped bar ──────────────
            if not cmp_df.empty and len(cmp_df) > 0:
                st.subheader("Detailed Metrics Breakdown")
                
                col_m1, col_m2 = st.columns(2, gap="large")
                
                with col_m1:
                    st.markdown("#### Accuracy Metrics (mAP, Precision, Recall)")
                    
                    fig_accuracy = go.Figure()
                    
                    for idx, row in cmp_df.iterrows():
                        model = row.get("Model", "Unknown")
                        color = {"YOLO": "#60A5FA", "Faster R-CNN": "#F472B6", "DETR": "#34D399"}.get(model, "#999999")
                        
                        def parse_metric(val):
                            if isinstance(val, str):
                                try:
                                    return float(val)
                                except:
                                    return 0.0
                            return float(val) if val else 0.0
                        
                        map50 = parse_metric(row.get("mAP @0.5", 0))
                        precision = parse_metric(row.get("Precision", 0))
                        recall = parse_metric(row.get("Recall", 0))
                        
                        fig_accuracy.add_trace(go.Bar(
                            name=model,
                            x=["mAP@0.5", "Precision", "Recall"],
                            y=[map50, precision, recall],
                            marker_color=color,
                            opacity=0.8,
                            text=[f"{v:.3f}" for v in [map50, precision, recall]],
                            textposition="outside",
                            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.4f}<extra></extra>",
                        ))
                    
                    fig_accuracy.update_layout(
                        barmode="group",
                        xaxis=dict(
                            title="",
                            tickfont=dict(size=9, family="Arial", color="#94A3B8"),
                            gridcolor="#1E293B",
                            showgrid=False,
                        ),
                        yaxis=dict(
                            title="Score",
                            range=[0, 1.15],
                            tickfont=dict(size=9, family="Arial", color="#94A3B8"),
                            gridcolor="#1E293B",
                            gridwidth=0.5,
                            title_font=dict(size=10, family="Arial", color="#94A3B8"),
                        ),
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=380,
                        font=dict(size=10, family="Arial", color="#E2E8F0"),
                        legend=dict(
                            orientation="h",
                            x=0.5,
                            y=1.12,
                            xanchor="center",
                            yanchor="top",
                            bgcolor="rgba(0,0,0,0)",
                            bordercolor="rgba(0,0,0,0)",
                            font=dict(size=9, family="Arial"),
                        ),
                        margin=dict(t=30, b=40, l=45, r=10),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_accuracy, use_container_width=True)
                
                with col_m2:
                    st.markdown("#### F1-Score & Speed Trade-off")
                    
                    fig_f1_speed = go.Figure()
                    
                    for idx, row in cmp_df.iterrows():
                        model = row.get("Model", "Unknown")
                        color = {"YOLO": "#60A5FA", "Faster R-CNN": "#F472B6", "DETR": "#34D399"}.get(model, "#999999")
                        
                        def parse_metric(val):
                            if isinstance(val, str):
                                try:
                                    return float(val)
                                except:
                                    return 0.0
                            return float(val) if val else 0.0
                        
                        f1 = parse_metric(row.get("F1-Score", 0))
                        latency_ms = parse_metric(row.get("Latency(ms)", 1000.0))
                        speed = max(0.1, 1.0 - min(latency_ms / 3000.0, 1.0))
                        
                        fig_f1_speed.add_trace(go.Bar(
                            name=model,
                            x=["F1-Score", "Speed"],
                            y=[f1, speed],
                            marker_color=color,
                            opacity=0.8,
                            text=[f"{v:.3f}" for v in [f1, speed]],
                            textposition="outside",
                            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.4f}<extra></extra>",
                        ))
                    
                    fig_f1_speed.update_layout(
                        barmode="group",
                        xaxis=dict(
                            title="",
                            tickfont=dict(size=9, family="Arial", color="#94A3B8"),
                            gridcolor="#1E293B",
                            showgrid=False,
                        ),
                        yaxis=dict(
                            title="Score",
                            range=[0, 1.15],
                            tickfont=dict(size=9, family="Arial", color="#94A3B8"),
                            gridcolor="#1E293B",
                            gridwidth=0.5,
                            title_font=dict(size=10, family="Arial", color="#94A3B8"),
                        ),
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=380,
                        font=dict(size=10, family="Arial", color="#E2E8F0"),
                        legend=dict(
                            orientation="h",
                            x=0.5,
                            y=1.12,
                            xanchor="center",
                            yanchor="top",
                            bgcolor="rgba(0,0,0,0)",
                            bordercolor="rgba(0,0,0,0)",
                            font=dict(size=9, family="Arial"),
                        ),
                        margin=dict(t=30, b=40, l=45, r=10),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_f1_speed, use_container_width=True)
                
                # Latency vs Accuracy scatter plot
                st.markdown("#### Latency vs Accuracy Trade-off")
                
                fig_tradeoff = go.Figure()
                
                for idx, row in cmp_df.iterrows():
                    model = row.get("Model", "Unknown")
                    color = {"YOLO": "#60A5FA", "Faster R-CNN": "#F472B6", "DETR": "#34D399"}.get(model, "#999999")
                    
                    def parse_metric(val):
                        if isinstance(val, str):
                            try:
                                return float(val)
                            except:
                                return 0.0
                        return float(val) if val else 0.0
                    
                    map50 = parse_metric(row.get("mAP @0.5", 0))
                    latency_ms = parse_metric(row.get("Latency(ms)", 1000.0))
                    params = parse_metric(row.get("Params(M)", 30.0))
                    
                    fig_tradeoff.add_trace(go.Scatter(
                        x=[latency_ms],
                        y=[map50],
                        mode="markers+text",
                        name=model,
                        marker=dict(
                            size=params/2,  # Size proportional to parameters
                            color=color,
                            opacity=0.7,
                            line=dict(width=2, color="white"),
                        ),
                        text=[model],
                        textposition="top center",
                        hovertemplate=f"<b>{model}</b><br>Latency: %{{x:.1f}} ms<br>mAP@0.5: %{{y:.4f}}<br>Params: {params:.1f}M<extra></extra>",
                    ))
                
                fig_tradeoff.update_layout(
                    xaxis=dict(
                        title="Latency (ms/img)",
                        type="log",
                        gridcolor="#1E293B",
                        gridwidth=0.5,
                        tickfont=dict(size=9, family="Arial", color="#94A3B8"),
                        title_font=dict(size=10, family="Arial", color="#94A3B8"),
                    ),
                    yaxis=dict(
                        title="mAP@0.5",
                        range=[0, 1.05],
                        gridcolor="#1E293B",
                        gridwidth=0.5,
                        tickfont=dict(size=9, family="Arial", color="#94A3B8"),
                        title_font=dict(size=10, family="Arial", color="#94A3B8"),
                    ),
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=420,
                    font=dict(size=10, family="Arial", color="#E2E8F0"),
                    hovermode="closest",
                    margin=dict(t=20, b=45, l=55, r=20),
                )
                
                st.plotly_chart(fig_tradeoff, use_container_width=True)
                
                st.caption("Bubble size = model parameters. Lower latency + higher mAP = better performance.")

            st.markdown("---")
            col_a1, col_a2 = st.columns(2, gap="large")

            with col_a1:
                st.subheader("Detections per Model")
                mc = raw_counts
                fig_mc = go.Figure(data=[go.Bar(
                    x=list(mc.keys()),
                    y=list(mc.values()),
                    marker_color=["#60A5FA", "#34D399", "#F472B6"],
                    text=list(mc.values()),
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Detections: %{y}<extra></extra>",
                )])
                fig_mc.update_layout(
                    xaxis=dict(
                        title="",
                        tickfont=dict(size=9, family="Arial", color="#94A3B8"),
                        showgrid=False,
                    ),
                    yaxis=dict(
                        title="Count",
                        tickfont=dict(size=9, family="Arial", color="#94A3B8"),
                        gridcolor="#1E293B",
                        gridwidth=0.5,
                        title_font=dict(size=10, family="Arial", color="#94A3B8"),
                    ),
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=320,
                    font=dict(size=10, family="Arial", color="#E2E8F0"),
                    margin=dict(t=10, b=35, l=45, r=10),
                    yaxis_rangemode="tozero",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_mc, use_container_width=True)

            with col_a2:
                st.subheader("Confidence Distribution")
                model_series = [
                    (result.yolo_raw, "YOLO", "#60A5FA"),
                    (result.detr_raw, "DETR", "#34D399"),
                    (result.rcnn_raw, "RCNN", "#F472B6"),
                ]
                if total_raw > 0:
                    fig_hist = go.Figure()
                    for data, name, color in model_series:
                        if data:
                            fig_hist.add_trace(go.Histogram(
                                x=[d.confidence for d in data],
                                name=name,
                                marker_color=color,
                                marker_line_width=0,
                                opacity=0.8,
                                nbinsx=8,
                                autobinx=False,
                                xbins=dict(start=0, end=1.0, size=0.125),
                                hovertemplate="<b>%{fullData.name}</b><br>Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>",
                            ))
                    fig_hist.update_layout(
                        barmode="overlay",
                        xaxis=dict(
                            title="Confidence Score",
                            range=[0, 1.05],
                            tickfont=dict(size=9, family="Arial", color="#94A3B8"),
                            title_font=dict(size=10, family="Arial", color="#94A3B8"),
                            showgrid=False,
                        ),
                        yaxis=dict(
                            title="Count",
                            tickfont=dict(size=9, family="Arial", color="#94A3B8"),
                            gridcolor="#1E293B",
                            gridwidth=0.5,
                            title_font=dict(size=10, family="Arial", color="#94A3B8"),
                        ),
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=360,
                        font=dict(size=9, family="Arial", color="#E2E8F0"),
                        legend=dict(
                            orientation="h",
                            x=0.5,
                            y=1.12,
                            xanchor="center",
                            yanchor="top",
                            bgcolor="rgba(0,0,0,0)",
                            bordercolor="rgba(0,0,0,0)",
                            font=dict(size=9, family="Arial"),
                        ),
                        margin=dict(t=30, b=35, l=45, r=10),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("No raw detections to visualise.")

            st.markdown("---")

            # ── Row B: Fused confidence bars  |  Model agreement matrix ─────────
            col_b1, col_b2 = st.columns([1.3, 1], gap="large")

            with col_b1:
                st.subheader("Fused Confidence by Constellation")
                if result.detections:
                    sorted_dets = sorted(result.detections, key=lambda d: d.confidence, reverse=True)
                    fig_conf = go.Figure(data=[go.Bar(
                        x=[d.label.replace("_", " ").title() for d in sorted_dets],
                        y=[d.confidence for d in sorted_dets],
                        marker_color=[
                            "#34D399" if d.confidence >= 0.7 else
                            "#FBBF24" if d.confidence >= 0.4 else "#F87171"
                            for d in sorted_dets
                        ],
                        text=[f"{d.confidence:.1%}" for d in sorted_dets],
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>",
                    )])
                    fig_conf.add_hline(y=0.7, line_dash="dash", line_color="#34D399", opacity=0.5)
                    fig_conf.add_hline(y=0.4, line_dash="dot", line_color="#FBBF24", opacity=0.5)
                    fig_conf.update_layout(
                        xaxis=dict(
                            title="",
                            tickfont=dict(size=8, family="Arial", color="#94A3B8"),
                            tickangle=-45,
                            showgrid=False,
                        ),
                        yaxis=dict(
                            title="Confidence",
                            range=[0, 1.25],
                            tickfont=dict(size=9, family="Arial", color="#94A3B8"),
                            gridcolor="#1E293B",
                            gridwidth=0.5,
                            title_font=dict(size=10, family="Arial", color="#94A3B8"),
                        ),
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=380,
                        font=dict(size=9, family="Arial", color="#E2E8F0"),
                        margin=dict(t=10, b=80, l=45, r=20),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
                else:
                    st.info("No fused detections to display.")

            with col_b2:
                st.subheader("Model Agreement Matrix")
                yolo_lbls  = {d.label for d in result.yolo_raw}
                detr_lbls  = {d.label for d in result.detr_raw}
                rcnn_lbls  = {d.label for d in result.rcnn_raw}
                fused_lbls = {d.label for d in result.detections}
                all_lbls   = sorted(yolo_lbls | detr_lbls | rcnn_lbls | fused_lbls)
                if all_lbls:
                    agree_rows = []
                    for lbl in all_lbls:
                        votes = sum([lbl in yolo_lbls, lbl in detr_lbls, lbl in rcnn_lbls])
                        agree_rows.append({
                            "Constellation": lbl.replace("_", " ").title(),
                            "YOLO":  "✅" if lbl in yolo_lbls  else "—",
                            "DETR":  "✅" if lbl in detr_lbls  else "—",
                            "RCNN":  "✅" if lbl in rcnn_lbls  else "—",
                            "Fused": "✅" if lbl in fused_lbls else "—",
                            "Votes": votes,
                        })
                    df_agree = pd.DataFrame(agree_rows).set_index("Constellation")
                    st.dataframe(df_agree, use_container_width=True, height=360)
                else:
                    st.info("No detections to compare.")

            # ── Per-model confidence grouped bar (only when multiple models ran) ─
            multi_model_dets = [d for d in [result.yolo_raw, result.detr_raw, result.rcnn_raw] if d]
            if len(multi_model_dets) > 1 and all_lbls:
                st.markdown("---")
                st.subheader("Per-Model Confidence by Label")
                # Group best confidence per label per model
                def best_conf(det_list, lbl):
                    matches = [d.confidence for d in det_list if d.label == lbl]
                    return max(matches) if matches else None

                fig_grp = go.Figure()
                for data, name, color in model_series:
                    if data:
                        vals  = [best_conf(data, lbl) for lbl in all_lbls]
                        texts = [f"{v:.1%}" if v is not None else "" for v in vals]
                        fig_grp.add_trace(go.Bar(
                            name=name,
                            x=[lbl.replace("_", " ").title() for lbl in all_lbls],
                            y=[v if v is not None else 0 for v in vals],
                            text=texts,
                            textposition="outside",
                            marker_color=color,
                            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.1%}<extra></extra>",
                        ))
                fig_grp.update_layout(
                    barmode="group",
                    xaxis=dict(
                        title="",
                        tickfont=dict(size=8, family="Arial", color="#94A3B8"),
                        tickangle=-45,
                        showgrid=False,
                    ),
                    yaxis=dict(
                        title="Confidence",
                        range=[0, 1.2],
                        tickfont=dict(size=9, family="Arial", color="#94A3B8"),
                        gridcolor="#1E293B",
                        gridwidth=0.5,
                        title_font=dict(size=10, family="Arial", color="#94A3B8"),
                    ),
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=400,
                    font=dict(size=9, family="Arial", color="#E2E8F0"),
                    legend=dict(
                        orientation="h",
                        x=0.5,
                        y=1.08,
                        xanchor="center",
                        yanchor="top",
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(0,0,0,0)",
                        font=dict(size=9, family="Arial"),
                    ),
                    margin=dict(t=20, b=80, l=45, r=10),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_grp, use_container_width=True)

            # ── Star scatter map ─────────────────────────────────────────────────
            if result.stars:
                st.markdown("---")
                st.subheader("Star Map")
                h_img = result.original_rgb.shape[0] if result.original_rgb is not None else 640
                w_img = result.original_rgb.shape[1] if result.original_rgb is not None else 640
                fig_stars = go.Figure(data=[go.Scatter(
                    x=[s.x for s in result.stars],
                    y=[h_img - s.y for s in result.stars],
                    mode="markers",
                    name="Stars",
                    marker=dict(
                        size=[max(4, min(s.sigma * 3, 20)) for s in result.stars],
                        color=[s.brightness for s in result.stars],
                        colorscale="Blues",
                        showscale=True,
                        colorbar=dict(title="Brightness", thickness=12, tickfont=dict(size=9, family="Arial")),
                        opacity=0.85,
                        line=dict(width=0.3, color="#1E293B"),
                    ),
                    text=[
                        f"Star @ ({s.x:.0f}, {s.y:.0f})<br>σ={s.sigma:.2f}, B={s.brightness:.3f}"
                        for s in result.stars
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )])
                # Overlay fused detection boxes
                for idx, det in enumerate(result.detections):
                    x1, y1, x2, y2 = det.bbox
                    r, g, b_ch = PALETTE[idx % len(PALETTE)]
                    hex_col = f"#{r:02x}{g:02x}{b_ch:02x}"
                    fig_stars.add_shape(
                        type="rect",
                        x0=x1, y0=h_img - y2, x1=x2, y1=h_img - y1,
                        line=dict(color=hex_col, width=2),
                    )
                    fig_stars.add_annotation(
                        x=(x1 + x2) / 2, y=h_img - y1 + 10,
                        text=det.label.replace("_", " ").title(),
                        showarrow=False,
                        font=dict(size=9, color=hex_col, family="Arial"),
                        bgcolor="rgba(0,0,0,0.6)",
                        bordercolor=hex_col,
                        borderwidth=1,
                        borderpad=3,
                    )
                fig_stars.update_layout(
                    xaxis=dict(
                        range=[0, w_img],
                        showgrid=True,
                        gridwidth=0.5,
                        gridcolor="#1E293B",
                        zeroline=False,
                        title="X (px)",
                        tickfont=dict(size=8, family="Arial", color="#94A3B8"),
                        title_font=dict(size=9, family="Arial", color="#94A3B8"),
                    ),
                    yaxis=dict(
                        range=[0, h_img],
                        showgrid=True,
                        gridwidth=0.5,
                        gridcolor="#1E293B",
                        zeroline=False,
                        title="Y (px)",
                        scaleanchor="x",
                        tickfont=dict(size=8, family="Arial", color="#94A3B8"),
                        title_font=dict(size=9, family="Arial", color="#94A3B8"),
                    ),
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=520,
                    font=dict(size=9, family="Arial", color="#E2E8F0"),
                    margin=dict(t=10, b=45, l=50, r=20),
                    hovermode="closest",
                )
                st.plotly_chart(fig_stars, use_container_width=True)
                st.caption(
                    f"{len(result.stars)} stars extracted. Marker size = blob scale | Color = brightness | Colored boxes = fused detections"
                )

            # ── Full detection dataframe ─────────────────────────────────────────
            st.markdown("---")
            st.subheader("📋 Complete Detection Table")
            det_rows = []
            fused_lbls_set = {d.label for d in result.detections}
            for model_name, det_list in [
                ("YOLO", result.yolo_raw),
                ("DETR", result.detr_raw),
                ("RCNN", result.rcnn_raw),
            ]:
                for det in det_list:
                    b = det.bbox
                    area = (b[2] - b[0]) * (b[3] - b[1])
                    det_rows.append({
                        "Model":         model_name,
                        "Constellation": det.label.replace("_", " ").title(),
                        "Confidence":    round(det.confidence, 4),
                        "x1": int(b[0]), "y1": int(b[1]),
                        "x2": int(b[2]), "y2": int(b[3]),
                        "W×H (px)":      f"{b[2]-b[0]:.0f}×{b[3]-b[1]:.0f}",
                        "Area (px²)":    int(area),
                        "In Fused":      det.label in fused_lbls_set,
                    })
            if det_rows:
                df_all = pd.DataFrame(det_rows)
                st.dataframe(df_all, use_container_width=True)
            else:
                st.info("No raw detections from any model.")

            # ── Collapsible extras ────────────────────────────────────────────────
            with st.expander("📄 Pipeline Summary Log", expanded=False):
                st.code(result.summarise())

            with st.expander(f"⭐ Star Data Table ({len(result.stars)} stars)", expanded=False):
                if result.stars:
                    star_data = [
                        {
                            "#": i, "X": f"{s.x:.1f}", "Y": f"{s.y:.1f}",
                            "Sigma": f"{s.sigma:.3f}", "Brightness": f"{s.brightness:.4f}",
                        }
                        for i, s in enumerate(result.stars[:50], 1)
                    ]
                    st.dataframe(pd.DataFrame(star_data), use_container_width=True)
                    if len(result.stars) > 50:
                        st.caption(f"Showing first 50 of {len(result.stars)} stars.")
                else:
                    st.info("No stars extracted.")

        else:
            st.info("▶ Run detection first (Detect tab) to populate the dashboard.")

    # ══ TAB 3: About ══════════════════════════════════════════════════════════
    with tab3:
        st.subheader("ℹ️ About Star AI")

        st.markdown("""
### 🌌 Hybrid Multi-Constellation Detection System

This application uses a **6-step hybrid pipeline** combining three AI detectors:

#### Pipeline Steps:
1. **📷 Preprocessing** — Letterbox resize to 640 × 640, CLAHE contrast enhancement, Gaussian denoising, float32 normalisation
2. **⭐ Star Extraction** — Blob detection (Laplacian of Gaussian) to locate stars
3. **🎯 YOLO Detection** — Fast single-pass object detection (YOLOv8m)
4. **🔍 DETR Detection** — Transformer-based global-context detection
5. **🔬 RCNN Detection** — Region-proposal network for precise localisation
6. **🔗 Result Fusion** — Multi-model agreement to produce a final verified list

---

#### 🤖 AI Models

**1️⃣ YOLO — You Only Look Once**
- Architecture: YOLOv8m fine-tuned on constellation images
- Strengths: Extremely fast; good for real-time single-pass detection

**2️⃣ DETR — DEtection TRansformer**
- Architecture: ResNet-50 + Transformer encoder-decoder (HuggingFace `facebook/detr-resnet-50`)
- Classes: 17 constellation classes (fine-tuned; default head is 88 — overridden by config)
- Strengths: Global attention captures large-scale star patterns; fewer duplicates

**3️⃣ RCNN — Region-based CNN (Faster R-CNN)**
- Architecture: ResNet-50-FPN + Region Proposal Network (torchvision)
- Strengths: High precision bounding boxes; strong two-stage detection
        """)

        # ── System Comparison (project-backed) ───────────────────────────────
        st.markdown("""
---

#### 📊 System Comparison (Project Data)
""")

        dataset_profile = load_dataset_profile()
        if dataset_profile["rows"]:
            st.markdown("**Dataset Profile**")
            st.dataframe(pd.DataFrame(dataset_profile["rows"]), use_container_width=True, hide_index=True)
            st.caption(
                f"Classes: {dataset_profile['class_count']} | "
                f"Total Images: {dataset_profile['total_images']} | "
                f"Total Annotations: {dataset_profile['total_annotations']}"
            )
        else:
            st.info("Dataset annotation stats were not found at data/constellation_dataset/annotations.")



        st.markdown("""
    **🔗 Fusion Engine**
    - Combines YOLO + DETR + RCNN detections by label + IoU overlap clustering
    - A detection is accepted if ≥ 2 models agree (IoU ≥ 0.40), or a single model
      exceeds the high-confidence threshold (≥ 0.85)

    ---

    #### 🎯 Detectable Constellations
    The deployed detectors recognize **17 primary constellation classes**.
    The training dataset covers **up to 88 IAU constellations**, but the active
    models are fine-tuned on the 17-class subset shown below.
        """)

        cols = st.columns(3)
        for i, (key, desc) in enumerate(CONSTELLATION_INFO.items()):
            with cols[i % 3]:
                label = key.replace("_", " ").title()
                st.markdown(
                    f"<div style='background:#1E293B;border-radius:8px;padding:10px 14px;"
                    f"margin-bottom:8px;border-left:3px solid #3B82F6;'>"
                    f"<div style='font-weight:700;color:#F1F5F9;'>✨ {label}</div>"
                    f"<div style='font-size:0.78rem;color:#64748B;margin-top:2px;'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("""
---

#### 🚀 How to Use
1. **Upload** a night sky image (PNG / JPG / BMP)
2. **Select** which detectors to run in the sidebar
3. Click **"Start Detection"** and wait for all models to complete
4. View each model's results in the expandable sections below
5. Check the **Fused Final Results** for the consensus output

#### 📝 Tips for Best Results
- Use high-resolution images (≥ 640 × 640 px)
- Ensure stars are clearly visible and not washed out
- Avoid heavy light pollution
- Enable all three models for the most accurate fused result

#### 🎨 Confidence Legend
- 🟢 **≥ 70 %** — High confidence detection
- 🟡 **40–69 %** — Medium confidence
- 🔴 **< 40 %** — Low confidence

---

**Built with:** PyTorch · Ultralytics YOLO · HuggingFace Transformers · torchvision · OpenCV · scikit-image · Plotly · Streamlit
        """)


if __name__ == "__main__":
    main()
