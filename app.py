"""
AquaScan — Sea Life Detection
- Correction memory: ORB visual similarity + class-name fallback
- Fine-tune: downloads original Kaggle dataset, merges corrections, retrains
- Credentials: kaggle.json next to app.py
"""

import streamlit as st
import numpy as np
import cv2
import json
import base64
import shutil
import yaml
import zipfile
import os
import subprocess
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import streamlit.elements.image as st_image
from streamlit.elements.lib.image_utils import image_to_url as _st_image_to_url
from streamlit.elements.lib.layout_utils import LayoutConfig

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

if not hasattr(st_image, "image_to_url"):
    # Streamlit 1.54+ moved this helper; keep canvas compatible across app versions.
    def _canvas_image_to_url(image, width, clamp, channels, output_format, image_id):
        layout_width = width if isinstance(width, int) else "content"
        return _st_image_to_url(
            image,
            layout_config=LayoutConfig(width=layout_width),
            clamp=clamp,
            channels=channels,
            output_format=output_format,
            image_id=image_id,
        )

    st_image.image_to_url = _canvas_image_to_url

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
MODEL_PATH       = BASE_DIR / "best (1).pt"
KAGGLE_JSON      = BASE_DIR / "kaggle.json"
CORRECTIONS_FILE = BASE_DIR / "corrections.json"
COMBINED_DIR     = BASE_DIR / "combined_dataset"
KAGGLE_CACHE     = BASE_DIR / "kaggle_dataset_cache"   # downloaded once, reused
RUNS_DIR         = BASE_DIR / "runs" / "finetune"

FINETUNE_THRESHOLD = 20
ORB_MATCH_RATIO    = 0.60
MIN_GOOD_MATCHES   = 12

# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AquaScan · Sea Life Detection",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0a0a0a;
    --surface:   #111111;
    --surface-2: #181818;
    --border:    #2a2a2a;
    --border-hi: #444444;
    --white:     #f5f5f0;
    --off-white: #c8c8c0;
    --muted:     #5a5a55;
    --danger:    #cc4444;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--white) !important;
    font-family: 'DM Mono', monospace !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }

/* HERO */
.hero {
    padding: 3rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
    display: flex; align-items: flex-end;
    justify-content: space-between;
    gap: 1rem; flex-wrap: wrap;
}
.hero-eyebrow {
    font-size: .65rem; letter-spacing: .22em;
    text-transform: uppercase; color: var(--muted); margin-bottom: .5rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 400; line-height: .92;
    color: var(--white); letter-spacing: -.02em;
}
.hero-title em { font-style: italic; color: var(--off-white); }
.hero-right {
    font-size: .68rem; color: var(--muted);
    text-align: right; line-height: 2; letter-spacing: .06em;
}

/* PANEL */
.panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 2px; padding: 1.6rem; margin-bottom: 1rem;
}
.panel-label {
    font-size: .62rem; letter-spacing: .2em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 1.2rem; padding-bottom: .7rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: .5rem;
}
.panel-label::before {
    content: ''; display: inline-block;
    width: 5px; height: 5px; background: var(--white); border-radius: 50%;
}

/* UPLOAD ZONE */
.upload-zone {
    border: 1px dashed var(--border-hi); border-radius: 2px;
    padding: 3.5rem 1.5rem; text-align: center;
}
.upload-glyph {
    font-family: 'DM Serif Display', serif; font-size: 3.5rem;
    color: var(--border-hi); display: block; margin-bottom: .8rem; line-height: 1;
}
.upload-hint { font-size: .68rem; color: var(--muted); letter-spacing: .1em; line-height: 2; }

/* PRIMARY BUTTON */
div.stButton > button {
    width: 100% !important; background: var(--white) !important;
    color: var(--bg) !important; font-family: 'DM Mono', monospace !important;
    font-size: .72rem !important; font-weight: 500 !important;
    letter-spacing: .14em !important; text-transform: uppercase !important;
    padding: .9rem 1.5rem !important; border: none !important;
    border-radius: 2px !important; cursor: pointer !important; transition: all .15s !important;
}
div.stButton > button:hover {
    background: var(--off-white) !important; transform: translateY(-1px) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }
div.stButton > button:disabled {
    background: var(--border) !important; color: var(--muted) !important;
    cursor: not-allowed !important; transform: none !important;
}

/* DELETE BUTTON */
.del-btn div.stButton > button {
    background: transparent !important; color: var(--muted) !important;
    border: 1px solid var(--border) !important; font-size: .65rem !important;
    padding: .25rem .55rem !important; height: 2rem !important;
    line-height: 1 !important; transform: none !important; text-transform: none !important;
}
.del-btn div.stButton > button:hover {
    border-color: var(--danger) !important; color: var(--danger) !important;
    background: rgba(204,68,68,.06) !important; transform: none !important;
}

/* SAVE BUTTON */
.save-btn div.stButton > button {
    background: transparent !important; color: var(--white) !important;
    border: 1px solid var(--border-hi) !important;
}
.save-btn div.stButton > button:hover {
    background: var(--white) !important; color: var(--bg) !important;
    border-color: var(--white) !important;
}

/* FINETUNE BUTTON */
.finetune-btn div.stButton > button {
    background: transparent !important; color: #e07070 !important;
    border: 1px solid rgba(204,68,68,.5) !important;
}
.finetune-btn div.stButton > button:hover {
    background: rgba(204,68,68,.08) !important;
    border-color: var(--danger) !important; color: #ff9090 !important; transform: none !important;
}
.finetune-btn div.stButton > button:disabled {
    background: transparent !important; color: var(--muted) !important;
    border-color: var(--border) !important; transform: none !important;
}

/* SLIDER */
.stSlider > div > div > div > div { background: var(--white) !important; }
.stSlider > div > div > div       { background: var(--border) !important; }
.stSlider label {
    font-family: 'DM Mono', monospace !important; font-size: .65rem !important;
    letter-spacing: .12em !important; text-transform: uppercase !important; color: var(--muted) !important;
}

/* TEXT / NUMBER INPUTS */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: 2px !important; color: var(--white) !important;
    font-family: 'DM Mono', monospace !important; font-size: .78rem !important;
    padding: .5rem .75rem !important; transition: border-color .15s !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--border-hi) !important; box-shadow: none !important;
}
.stTextInput label, .stNumberInput label {
    font-family: 'DM Mono', monospace !important; font-size: .65rem !important;
    letter-spacing: .1em !important; text-transform: uppercase !important; color: var(--muted) !important;
}

/* FILE UPLOADER */
.stFileUploader > div {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}
.stFileUploader label {
    font-family: 'DM Mono', monospace !important; font-size: .65rem !important; color: var(--muted) !important;
}

/* STATS */
.stats-row {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: .5rem; margin-bottom: 1.5rem;
}
.stat-box {
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 2px; padding: 1rem .8rem; text-align: center;
}
.stat-val { font-family: 'DM Serif Display', serif; font-size: 2rem; color: var(--white); line-height: 1; }
.stat-key { font-size: .58rem; text-transform: uppercase; letter-spacing: .16em; color: var(--muted); margin-top: .35rem; }

/* DET ROWS */
.det-num  { font-size: .62rem; color: var(--muted); padding-top: .55rem; }
.det-conf { font-size: .65rem; color: var(--muted); text-align: right; padding-top: .55rem; }
.det-badge {
    font-size: .58rem; padding: .12rem .45rem; border-radius: 2px;
    border: 1px solid var(--border-hi); color: var(--muted);
    white-space: nowrap; margin-top: .4rem; display: inline-block;
}
.det-badge.visual { border-color: rgba(100,180,255,.3); color: #6ab4ff; }
.det-badge.name   { border-color: rgba(160,255,160,.3); color: #90d090; }

/* LABEL INPUT */
.det-input div[data-testid="stTextInput"] { margin: 0 !important; }
.det-input div[data-testid="stTextInput"] > label { display: none !important; }
.det-input div[data-testid="stTextInput"] > div > div > input {
    background: var(--bg) !important; border: 1px solid var(--border-hi) !important;
    border-radius: 2px !important; color: var(--white) !important;
    font-family: 'DM Mono', monospace !important; font-size: .72rem !important;
    padding: .28rem .6rem !important; height: 2rem !important; transition: border-color .15s !important;
}
.det-input div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--white) !important; box-shadow: none !important;
}

/* MEMORY PANEL */
.memory-panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 2px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.memory-count { font-family: 'DM Serif Display', serif; font-size: 2.4rem; color: var(--white); line-height: 1; }
.memory-sub   { font-size: .62rem; color: var(--muted); letter-spacing: .12em; text-transform: uppercase; margin-top: .2rem; }
.progress-track {
    height: 3px; background: var(--border); border-radius: 2px; margin: .8rem 0 .4rem; overflow: hidden;
}
.progress-fill { height: 100%; background: var(--white); border-radius: 2px; transition: width .4s ease; }
.progress-label { font-size: .6rem; color: var(--muted); letter-spacing: .1em; }

/* KAGGLE SETUP PANEL */
.setup-panel {
    background: var(--surface-2); border: 1px solid rgba(204,68,68,.3);
    border-radius: 2px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.setup-title {
    font-size: .62rem; letter-spacing: .2em; text-transform: uppercase;
    color: #e07070; margin-bottom: .8rem; display: flex; align-items: center; gap: .5rem;
}
.setup-hint { font-size: .68rem; color: var(--muted); line-height: 1.8; margin-bottom: .8rem; }

/* MESSAGES */
.msg-ok {
    background: rgba(245,245,240,.04); border: 1px solid var(--border-hi);
    border-radius: 2px; padding: .75rem 1rem; font-size: .7rem;
    color: var(--white); letter-spacing: .05em; margin-top: .8rem; line-height: 1.6;
}
.msg-warn {
    background: rgba(204,68,68,.05); border: 1px solid rgba(204,68,68,.3);
    border-radius: 2px; padding: .75rem 1rem; font-size: .7rem;
    color: #e07070; letter-spacing: .05em; margin-top: .8rem; line-height: 1.6;
}
.msg-info {
    background: rgba(100,180,255,.04); border: 1px solid rgba(100,180,255,.2);
    border-radius: 2px; padding: .75rem 1rem; font-size: .7rem;
    color: #6ab4ff; letter-spacing: .05em; margin-top: .8rem; line-height: 1.6;
}

/* RULE */
.rule { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }

/* EMPTY */
.empty { text-align: center; padding: 5rem 2rem; }
.empty-glyph {
    font-family: 'DM Serif Display', serif; font-size: 5rem;
    color: var(--border-hi); display: block; margin-bottom: 1rem; line-height: 1;
}
.empty p { font-size: .72rem; color: var(--muted); line-height: 2; letter-spacing: .06em; }

/* SPINNER */
.stSpinner > div { border-top-color: var(--white) !important; }

/* FOOTER */
.footer {
    text-align: center; padding: 2.5rem 0 1rem; font-size: .6rem;
    color: var(--muted); letter-spacing: .16em; text-transform: uppercase;
    border-top: 1px solid var(--border); margin-top: 3rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp .35s ease forwards; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Kaggle config helpers
# ══════════════════════════════════════════════

def load_kaggle_config() -> dict | None:
    """Load kaggle.json → {username, key, dataset_slug}"""
    if KAGGLE_JSON.exists():
        try:
            return json.loads(KAGGLE_JSON.read_text())
        except Exception:
            return None
    return None


def save_kaggle_config(username: str, key: str, slug: str):
    """
    Save to kaggle.json (same format Kaggle CLI expects, plus dataset_slug).
    Also write ~/.kaggle/kaggle.json so the CLI picks it up automatically.
    """
    cfg = {"username": username, "key": key, "dataset_slug": slug}
    KAGGLE_JSON.write_text(json.dumps(cfg, indent=2))
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    cli_cfg = kaggle_dir / "kaggle.json"
    cli_cfg.write_text(json.dumps({"username": username, "key": key}, indent=2))
    cli_cfg.chmod(0o600)


def setup_kaggle_env(cfg: dict):
    """Set env vars so kaggle CLI works for this process."""
    os.environ["KAGGLE_USERNAME"] = cfg["username"]
    os.environ["KAGGLE_KEY"]      = cfg["key"]


# ══════════════════════════════════════════════
# Kaggle dataset download
# ══════════════════════════════════════════════

def download_kaggle_dataset(slug: str, dest: Path) -> tuple[bool, str]:
    """
    Download and unzip a Kaggle dataset into dest/.
    Returns (success, message).
    Skips download if dest already contains images (cached).
    """
    # Check cache — if images folder already has files, skip download
    existing_images = list(dest.rglob("*.jpg")) + list(dest.rglob("*.png"))
    if len(existing_images) > 10:
        return True, f"Using cached dataset ({len(existing_images)} images found)."

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "dataset.zip"

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug,
             "-p", str(dest), "--unzip"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return False, result.stderr.strip() or "Kaggle download failed."
        return True, "Dataset downloaded successfully."
    except FileNotFoundError:
        # Try installing kaggle CLI if missing
        subprocess.run(["pip", "install", "kaggle", "-q"], capture_output=True)
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug,
                 "-p", str(dest), "--unzip"],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                return False, result.stderr.strip() or "Kaggle download failed."
            return True, "Dataset downloaded successfully."
        except Exception as e:
            return False, str(e)
    except subprocess.TimeoutExpired:
        return False, "Download timed out after 10 minutes."
    except Exception as e:
        return False, str(e)


# ══════════════════════════════════════════════
# Dataset merging
# ══════════════════════════════════════════════

def find_yolo_split(root: Path) -> tuple[Path | None, Path | None, Path | None]:
    """
    Search root for a YOLO-structured dataset.
    Returns (images_dir, labels_dir, data_yaml) or (None, None, None).
    Handles nested structures like root/train/images, root/images/train, etc.
    """
    # Find data.yaml first
    yaml_files = list(root.rglob("data.yaml"))
    data_yaml  = yaml_files[0] if yaml_files else None

    # Find images folder — prefer 'train' split
    for pattern in ["**/images/train", "**/train/images", "**/images"]:
        matches = list(root.glob(pattern))
        if matches:
            img_dir = matches[0]
            # Derive labels dir by replacing 'images' with 'labels'
            lbl_dir = Path(str(img_dir).replace("images", "labels"))
            if lbl_dir.exists():
                return img_dir, lbl_dir, data_yaml
            # labels at same level
            lbl_dir2 = img_dir.parent / "labels"
            if lbl_dir2.exists():
                return img_dir, lbl_dir2, data_yaml

    return None, None, data_yaml


def build_combined_dataset(kaggle_root: Path,
                            corrections: list,
                            out_dir: Path) -> tuple[Path | None, str]:
    """
    Merge original Kaggle dataset + corrections into out_dir.
    Returns (data_yaml_path, status_message).
    """
    out_img = out_dir / "images" / "train"
    out_lbl = out_dir / "labels"  / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # ── 1. Copy original dataset ──
    orig_img, orig_lbl, orig_yaml = find_yolo_split(kaggle_root)

    if orig_img is None:
        return None, ("Could not find images/ folder in the downloaded dataset. "
                      "Please check the dataset structure.")

    # Read original class names from data.yaml
    orig_names: list = []
    if orig_yaml and orig_yaml.exists():
        try:
            raw = yaml.safe_load(orig_yaml.read_text()).get("names", {})
            orig_names = list(raw.values()) if isinstance(raw, dict) else list(raw)
        except Exception:
            pass

    n_orig = 0
    for img_file in orig_img.iterdir():
        if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            shutil.copy2(str(img_file), str(out_img / img_file.name))
            lbl_file = orig_lbl / (img_file.stem + ".txt")
            if lbl_file.exists():
                shutil.copy2(str(lbl_file), str(out_lbl / lbl_file.name))
            n_orig += 1

    # ── 2. Add corrections ──
    # Corrections are full-image crops — we write them as full-frame YOLO labels
    # Only corrections with non-unknown, non-empty labels are included
    valid_corrections = [
        c for c in corrections
        if c["corrected_label"].strip()
        and c["corrected_label"].strip().lower() != "unknown"
    ]

    # Build unified class list: original names first, then any new correction classes
    corr_classes = list(dict.fromkeys(
        c["corrected_label"].strip() for c in valid_corrections))
    all_classes   = list(dict.fromkeys(orig_names + corr_classes))
    label_to_id   = {n: i for i, n in enumerate(all_classes)}

    # Re-write original labels with updated class IDs if names match
    # (handles case where original yaml had same classes under different IDs)
    orig_id_map: dict = {}
    if orig_names:
        for i, name in enumerate(orig_names):
            if name in label_to_id:
                orig_id_map[i] = label_to_id[name]

    if orig_id_map and orig_id_map != {i: i for i in orig_id_map}:
        for lbl_file in out_lbl.iterdir():
            if lbl_file.suffix == ".txt":
                lines     = lbl_file.read_text().strip().splitlines()
                new_lines = []
                for line in lines:
                    parts = line.split()
                    if parts:
                        old_id = int(parts[0])
                        new_id = orig_id_map.get(old_id, old_id)
                        new_lines.append(f"{new_id} " + " ".join(parts[1:]))
                lbl_file.write_text("\n".join(new_lines))

    n_corr = 0
    for entry in valid_corrections:
        crop_np = b64_to_np(entry["crop_b64"])
        if crop_np is None:
            continue
        stem     = f"correction_{entry['id']}"
        img_path = out_img / f"{stem}.jpg"
        cv2.imwrite(str(img_path), crop_np)

        cid  = label_to_id[entry["corrected_label"].strip()]
        # bbox = full crop frame
        (out_lbl / f"{stem}.txt").write_text(f"{cid} 0.5 0.5 1.0 1.0")
        n_corr += 1

    # ── 3. Write combined data.yaml ──
    out_yaml = out_dir / "data.yaml"
    out_yaml.write_text(yaml.dump({
        "path":  str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/train",
        "names": {i: n for i, n in enumerate(all_classes)},
        "nc":    len(all_classes),
    }, default_flow_style=False))

    msg = (f"Combined dataset ready: {n_orig} original images "
           f"+ {n_corr} corrections = {n_orig + n_corr} total.")
    return out_yaml, msg


# ══════════════════════════════════════════════
# Fine-tune
# ══════════════════════════════════════════════

def run_finetune(data_yaml: Path, epochs: int = 20) -> str:
    """
    Fine-tune on combined dataset (original + corrections).
    Uses moderate freeze so backbone is preserved, only detection head adapts.
    Overwrites MODEL_PATH with best weights.
    """
    try:
        # Clean previous run so we don't pick up stale weights
        if RUNS_DIR.exists():
            shutil.rmtree(str(RUNS_DIR))

        model = YOLO(str(MODEL_PATH))
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            batch=8,
            lr0=1e-4,       # conservative LR
            lrf=0.01,        # final LR factor
            warmup_epochs=3,
            freeze=10,       # freeze first 10 backbone layers
            project=str(RUNS_DIR.parent),
            name=RUNS_DIR.name,
            exist_ok=True,
            verbose=False,
            plots=False,
        )
        best = RUNS_DIR / "weights" / "best.pt"
        if best.exists():
            shutil.copy(str(best), str(MODEL_PATH))
            st.cache_resource.clear()
            return "ok"
        return "Training finished but best.pt not found."
    except Exception as e:
        return str(e)


# ══════════════════════════════════════════════
# Correction memory
# ══════════════════════════════════════════════

def normalize_correction_entry(entry: object, fallback_id: str) -> dict | None:
    if not isinstance(entry, dict):
        return None

    label = str(entry.get("corrected_label") or entry.get("label") or "").strip()
    if not label:
        return None

    return {
        "id": str(entry.get("id") or fallback_id),
        "corrected_label": label,
        "original_class": str(
            entry.get("original_class") or entry.get("original_label") or ""
        ),
        "crop_b64": str(entry.get("crop_b64") or entry.get("image_b64") or ""),
        "timestamp": str(entry.get("timestamp") or ""),
    }


def load_corrections() -> list:
    if not CORRECTIONS_FILE.exists():
        return []

    try:
        raw = json.loads(CORRECTIONS_FILE.read_text())
    except Exception:
        return []

    if not isinstance(raw, list):
        return []

    cleaned = []
    for idx, entry in enumerate(raw):
        normalized = normalize_correction_entry(entry, fallback_id=f"legacy_{idx:04d}")
        if normalized is not None:
            cleaned.append(normalized)

    if cleaned != raw:
        try:
            save_corrections(cleaned)
        except Exception:
            pass

    return cleaned


def save_corrections(corrections: list):
    cleaned = []
    for idx, entry in enumerate(corrections):
        normalized = normalize_correction_entry(
            entry, fallback_id=f"correction_{idx:04d}"
        )
        if normalized is not None:
            cleaned.append(normalized)
    CORRECTIONS_FILE.write_text(json.dumps(cleaned, indent=2))


def crop_to_b64(img_np: np.ndarray, bbox: list) -> str:
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    x2 = min(img_np.shape[1], x2)
    y2 = min(img_np.shape[0], y2)
    crop = img_np[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    buf = BytesIO()
    Image.fromarray(crop).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_np(b64: str) -> np.ndarray | None:
    try:
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_orb():
    return cv2.ORB_create(nfeatures=500)


def orb_similarity(crop_np: np.ndarray, stored_np: np.ndarray) -> int:
    orb = get_orb()
    stored_r = cv2.resize(stored_np, (max(1, crop_np.shape[1]),
                                       max(1, crop_np.shape[0])))
    kp1, des1 = orb.detectAndCompute(crop_np,  None)
    kp2, des2 = orb.detectAndCompute(stored_r, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    return sum(1 for m, n in matches if m.distance < ORB_MATCH_RATIO * n.distance)


def apply_correction_memory(img_np: np.ndarray, detections: list,
                             corrections: list) -> tuple[list, list]:
    if not corrections:
        return detections, [None] * len(detections)

    match_sources = []
    for i, det in enumerate(detections):
        raw_b64 = crop_to_b64(img_np, det["bbox"])
        crop_np = b64_to_np(raw_b64) if raw_b64 else None

        best_visual_label = None
        best_visual_score = 0

        for entry in corrections:
            stored_label = str(entry.get("corrected_label", "")).strip()
            if not stored_label:
                continue
            stored_orig  = entry.get("original_class", "")

            # Visual
            if crop_np is not None and entry.get("crop_b64"):
                stored_np = b64_to_np(entry["crop_b64"])
                if stored_np is not None:
                    score = orb_similarity(crop_np, stored_np)
                    if score > best_visual_score:
                        best_visual_score = score
                        best_visual_label = stored_label

            # Name fallback
            if best_visual_score >= MIN_GOOD_MATCHES:
                detections[i]["label"] = best_visual_label
                match_sources.append("visual")
            else:
                match_sources.append(None)

    return detections, match_sources


# ══════════════════════════════════════════════
# Model & rendering
# ══════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(str(MODEL_PATH))


def render_preview(img_np: np.ndarray, detections: list) -> np.ndarray:
    out    = img_np.copy()
    h, w   = out.shape[:2]
    scale  = max(w, h) / 1000
    fscale = max(0.45, min(1.3, scale))
    thick  = max(2, int(scale * 2.2))

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        is_unk    = det["label"].strip().lower() in ("", "unknown")
        box_color = (140, 140, 135) if is_unk else (245, 245, 240)

        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
        cv2.addWeighted(overlay, 0.08, out, 0.92, 0, out)
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, thick)

        label_text = f"#{idx+1} {det['label'].strip() or 'Unknown'}  {det['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, fscale, max(1, thick - 1))
        pad  = max(6, int(scale * 6))
        py1  = max(0, y1 - th - pad * 2)
        py2  = py1 + th + pad * 2

        pill_bg = (55, 55, 55)    if is_unk else (245, 245, 240)
        txt_col = (200, 200, 195) if is_unk else (10, 10, 10)

        cv2.rectangle(out, (x1, py1), (x1 + tw + pad * 2, py2), pill_bg, -1)
        cv2.putText(out, label_text, (x1 + pad, py2 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale,
                    txt_col, max(1, thick - 1), cv2.LINE_AA)

    return out


# ══════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div>
        <div class="hero-eyebrow">◼ Marine Computer Vision</div>
        <div class="hero-title">Aqua<em>Scan</em></div>
    </div>
    <div class="hero-right">Sea Life Detection · YOLOv8</div>
</div>
""", unsafe_allow_html=True)

corrections   = load_corrections()
n_corrections = len(corrections)
kaggle_cfg    = load_kaggle_config()

col_left, _sp, col_right = st.columns([4, 0.2, 6])

# ══════════════════════════════════════════════
# LEFT COLUMN
# ══════════════════════════════════════════════
with col_left:

    # ── Kaggle setup (shown only when not configured) ──
    if kaggle_cfg is None:
        st.markdown("""
        <div class="setup-panel">
            <div class="setup-title">⚠ Kaggle Setup Required</div>
            <div class="setup-hint">
                Enter your Kaggle credentials once. These are saved to
                <code>kaggle.json</code> next to app.py and never sent anywhere.<br><br>
                Find your API key at kaggle.com → Settings → API → Create New Token.
            </div>
        </div>""", unsafe_allow_html=True)

        kg_user  = st.text_input("Kaggle Username", placeholder="your-username")
        kg_key   = st.text_input("Kaggle API Key",  placeholder="xxxxxxxxxxxxxxxx", type="password")
        kg_slug  = st.text_input("Dataset Slug",    placeholder="username/dataset-name",
                                  help="e.g. crowww/underwater-fish-detection")

        if st.button("◼  Save Kaggle Config", use_container_width=True,
                     disabled=not (kg_user and kg_key and kg_slug)):
            save_kaggle_config(kg_user.strip(), kg_key.strip(), kg_slug.strip())
            st.rerun()

        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    else:
        # Show slug + option to reset
        st.markdown(
            f'<div class="msg-ok" style="margin-bottom:.8rem">'
            f'◈ Kaggle configured · <code>{kaggle_cfg.get("dataset_slug","")}</code></div>',
            unsafe_allow_html=True)
        if st.button("↺  Reset Kaggle Config", use_container_width=True):
            KAGGLE_JSON.unlink(missing_ok=True)
            st.rerun()

    # ── Upload ──
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Upload Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "image", type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="upload-zone">
            <span class="upload-glyph">◈</span>
            <div class="upload-hint">JPG · PNG · WEBP · BMP<br>drag & drop or click to browse</div>
        </div>""", unsafe_allow_html=True)
        for k in ("detections", "img_np", "pil_img", "match_sources", "save_status"):
            st.session_state.pop(k, None)
    else:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, use_container_width=True)
        st.session_state["pil_img"] = pil_img

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Settings ──
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Settings</div>', unsafe_allow_html=True)

    threshold = st.slider(
        "Unknown confidence threshold",
        0.10, 0.95, 0.50, 0.05, format="%.2f",
        help="Detections below this confidence are labelled UNKNOWN",
    )

    run_btn = st.button(
        "◼  Identify Sea Life",
        use_container_width=True,
        disabled=(uploaded_file is None),
    )

    # Save Correction button
    has_dets = ("detections" in st.session_state
                and len(st.session_state["detections"]) > 0)
    if has_dets:
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:.68rem;color:var(--muted);line-height:1.8;margin-bottom:.9rem">'
            'Edit labels · remove wrong boxes · then save to correction memory.</p>',
            unsafe_allow_html=True)
        st.markdown('<div class="save-btn">', unsafe_allow_html=True)
        save_btn = st.button("◈  Save Corrections", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        save_btn = False

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Correction memory counter ──
    pct = min(100, int(n_corrections / FINETUNE_THRESHOLD * 100))
    st.markdown(f"""
    <div class="memory-panel">
        <div class="panel-label">Correction Memory</div>
        <div class="memory-count">{n_corrections}</div>
        <div class="memory-sub">corrections stored</div>
        <div class="progress-track">
            <div class="progress-fill" style="width:{pct}%"></div>
        </div>
        <div class="progress-label">{n_corrections} / {FINETUNE_THRESHOLD} to unlock fine-tuning</div>
    </div>""", unsafe_allow_html=True)

    if "save_status" in st.session_state:
        st.markdown(
            f'<div class="msg-ok">✓ {st.session_state["save_status"]} correction(s) added.</div>',
            unsafe_allow_html=True)

    # ── Fine-tune (unlocks at threshold, requires Kaggle config) ──
    if n_corrections >= FINETUNE_THRESHOLD:
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="msg-warn">
            ⚠ Fine-tuning will:<br>
            1. Download your original Kaggle dataset (cached after first run)<br>
            2. Merge it with all saved corrections<br>
            3. Retrain the model on the combined data<br>
            4. Overwrite best (1).pt with new weights<br><br>
            This is safe because the original data is always included.
        </div>""", unsafe_allow_html=True)

        kaggle_ready = kaggle_cfg is not None
        if not kaggle_ready:
            st.markdown(
                '<div class="msg-warn" style="margin-top:.5rem">'
                '✕ Configure Kaggle credentials above first.</div>',
                unsafe_allow_html=True)

        st.markdown('<div class="finetune-btn" style="margin-top:.8rem">', unsafe_allow_html=True)
        finetune_btn = st.button(
            "⟳  Fine-Tune Model",
            use_container_width=True,
            disabled=not kaggle_ready,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if "train_status" in st.session_state:
            ts  = st.session_state["train_status"]
            cls = "msg-ok" if ts == "ok" else "msg-warn"
            msg = ("✓ Model updated successfully. "
                   "Run Identify Sea Life again to use new weights.") if ts == "ok" else f"✕ {ts}"
            st.markdown(f'<div class="{cls}" style="margin-top:.6rem">{msg}</div>',
                        unsafe_allow_html=True)
    else:
        finetune_btn = False


# ══════════════════════════════════════════════
# RIGHT COLUMN
# ══════════════════════════════════════════════
with col_right:
    st.markdown('<div class="panel fade-up">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Detection Results</div>', unsafe_allow_html=True)

    # ── Inference ──
    if run_btn and uploaded_file is not None:
        with st.spinner("Analysing image…"):
            model       = load_model()
            class_names = model.names
            img_np      = np.array(pil_img)
            results     = model.predict(img_np, conf=0.10, verbose=False)
            boxes       = results[0].boxes

        detections = []
        for box in boxes:
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            orig = class_names.get(cls_id, f"Class {cls_id}")
            detections.append({
                "cls_id":         cls_id,
                "conf":           conf_val,
                "bbox":           [x1, y1, x2, y2],
                "original_class": orig,
                "label":          "Unknown" if conf_val < threshold else orig,
            })

        with st.spinner("Checking correction memory…"):
            detections, match_sources = apply_correction_memory(
                img_np, detections, corrections)

        st.session_state["detections"]    = detections
        st.session_state["img_np"]        = img_np
        st.session_state["match_sources"] = match_sources
        st.session_state.pop("save_status",  None)
        st.session_state.pop("train_status", None)
        st.session_state["draw_mode"]          = False
        st.session_state["canvas_known_count"] = 0
        st.rerun()

    # ── Save corrections ──
    if save_btn and has_dets:
        img_np     = st.session_state.get("img_np")
        detections = st.session_state["detections"]
        saved      = 0
        for det in detections:
            label = det["label"].strip()
            if not label or label.lower() == "unknown":
                continue
            corrections.append({
                "id":              datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                "corrected_label": label,
                "original_class":  det.get("original_class", ""),
                "crop_b64":        crop_to_b64(img_np, det["bbox"]) if img_np is not None else "",
                "timestamp":       datetime.now().isoformat(),
            })
            saved += 1
        save_corrections(corrections)
        n_corrections = len(corrections)
        st.session_state["save_status"] = saved
        st.rerun()

    # ── Fine-tune ──
    if finetune_btn and kaggle_cfg:
        setup_kaggle_env(kaggle_cfg)
        slug = kaggle_cfg["dataset_slug"]

        # Step 1: Download
        with st.spinner(f"Downloading Kaggle dataset '{slug}' (cached after first run)…"):
            ok, msg = download_kaggle_dataset(slug, KAGGLE_CACHE)

        if not ok:
            st.session_state["train_status"] = f"Kaggle download failed: {msg}"
            st.rerun()

        # Step 2: Build combined dataset
        with st.spinner("Merging original dataset with corrections…"):
            if COMBINED_DIR.exists():
                shutil.rmtree(str(COMBINED_DIR))
            data_yaml, merge_msg = build_combined_dataset(
                KAGGLE_CACHE, corrections, COMBINED_DIR)

        if data_yaml is None:
            st.session_state["train_status"] = merge_msg
            st.rerun()

        # Step 3: Retrain
        with st.spinner(f"Fine-tuning on combined dataset — {merge_msg} Please wait…"):
            status = run_finetune(data_yaml, epochs=20)

        st.session_state["train_status"] = status
        st.rerun()

    # ── Display ──
    if "detections" in st.session_state:
        detections    = st.session_state["detections"]
        img_np        = st.session_state.get("img_np")
        match_sources = st.session_state.get("match_sources", [None] * len(detections))

        if len(detections) == 0:
            st.markdown("""
            <div class="empty">
                <span class="empty-glyph">◎</span>
                <p>No detections found.<br>Try lowering the threshold.</p>
            </div>""", unsafe_allow_html=True)
        else:
            identified = [d for d in detections
                          if d["label"].strip().lower() not in ("", "unknown")]
            unknowns   = [d for d in detections
                          if d["label"].strip().lower() in ("", "unknown")]
            avg_conf   = sum(d["conf"] for d in detections) / len(detections)

            st.markdown(f"""
            <div class="stats-row">
                <div class="stat-box"><div class="stat-val">{len(identified)}</div><div class="stat-key">Identified</div></div>
                <div class="stat-box"><div class="stat-val">{len(unknowns)}</div><div class="stat-key">Unknown</div></div>
                <div class="stat-box"><div class="stat-val">{len(set(d['label'] for d in identified))}</div><div class="stat-key">Species</div></div>
                <div class="stat-box"><div class="stat-val">{avg_conf:.0%}</div><div class="stat-key">Avg Conf</div></div>
            </div>""", unsafe_allow_html=True)

        # ── Canvas preview (always shown when img_np exists) ──
        if img_np is not None:
            # Render current detections onto the image as the canvas background
            annotated     = render_preview(img_np, detections)
            annotated_pil = Image.fromarray(annotated)

            # Fix canvas width to column width (~640px is reliable across screens)
            CANVAS_W = 640
            orig_h, orig_w = img_np.shape[:2]
            canvas_h = int(CANVAS_W * orig_h / orig_w)

            # Scale factor: canvas coords → original image coords
            scale_x = orig_w / CANVAS_W
            scale_y = orig_h / canvas_h

            # Draw mode toggle
            draw_mode = st.session_state.get("draw_mode", False)
            mode_label = "◼ Stop Drawing" if draw_mode else "◈ Draw New Box"
            mode_cls   = "save-btn" if draw_mode else "save-btn"

            st.markdown(
                f'<p style="font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;'
                f'color:var(--muted);margin-bottom:.5rem">'
                f'{"Click and drag on the image to draw a box" if draw_mode else "Switch to draw mode to add a missing detection"}'
                f'</p>',
                unsafe_allow_html=True)

            tog_col, _ = st.columns([2, 5])
            with tog_col:
                st.markdown(f'<div class="{mode_cls}">', unsafe_allow_html=True)
                if st.button(mode_label, key="toggle_draw", use_container_width=True):
                    st.session_state["draw_mode"] = not draw_mode
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            if CANVAS_AVAILABLE:
                canvas_result = st_canvas(
                    fill_color   = "rgba(245,245,240,0.08)",
                    stroke_width = 2,
                    stroke_color = "#f5f5f0",
                    background_image = annotated_pil,
                    update_streamlit = True,
                    width    = CANVAS_W,
                    height   = canvas_h,
                    drawing_mode = "rect" if draw_mode else "transform",
                    key      = "canvas",
                )

                # Pick up newly drawn rectangles
                if (canvas_result is not None
                        and canvas_result.json_data is not None
                        and draw_mode):
                    objects = canvas_result.json_data.get("objects", [])
                    # Only process rectangles not yet in session
                    n_known = st.session_state.get("canvas_known_count", 0)
                    new_rects = [o for o in objects if o.get("type") == "rect"]

                    if len(new_rects) > n_known:
                        # New box was just drawn — grab the last one
                        rect = new_rects[-1]
                        # Canvas gives left/top + width/height (canvas space)
                        cx1 = rect.get("left", 0)
                        cy1 = rect.get("top",  0)
                        cw  = rect.get("width",  0)
                        ch  = rect.get("height", 0)

                        # Convert to original image coordinates
                        x1 = max(0, int(cx1 * scale_x))
                        y1 = max(0, int(cy1 * scale_y))
                        x2 = min(orig_w, int((cx1 + cw) * scale_x))
                        y2 = min(orig_h, int((cy1 + ch) * scale_y))

                        if x2 > x1 + 5 and y2 > y1 + 5:   # ignore accidental tiny clicks
                            st.session_state["detections"].append({
                                "cls_id":         -1,
                                "conf":           1.0,
                                "bbox":           [x1, y1, x2, y2],
                                "original_class": "manual",
                                "label":          "Unknown",
                            })
                            st.session_state.setdefault("match_sources", []).append(None)
                            st.session_state["canvas_known_count"] = len(new_rects)
                            # Exit draw mode after adding so user can edit label
                            st.session_state["draw_mode"] = False
                            st.rerun()

                        st.session_state["canvas_known_count"] = len(new_rects)
            else:
                # Fallback: just show static image if canvas not installed
                st.image(annotated_pil, use_container_width=True)
                st.markdown(
                    '<div class="msg-warn">Install streamlit-drawable-canvas to enable box drawing.</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

        if len(detections) > 0:
            to_delete = []
            for i, det in enumerate(detections):
                src = match_sources[i] if i < len(match_sources) else None
                c_num, c_inp, c_conf, c_del = st.columns([0.55, 5, 1, 0.65])

                c_num.markdown(f'<div class="det-num">#{i+1}</div>', unsafe_allow_html=True)

                with c_inp:
                    st.markdown('<div class="det-input">', unsafe_allow_html=True)
                    new_label = st.text_input(
                        f"lbl_{i}", value=det["label"],
                        key=f"lbl_{i}", label_visibility="collapsed",
                        placeholder="Edit label…",
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    if src == "visual":
                        st.markdown('<span class="det-badge visual">◈ visual match</span>',
                                    unsafe_allow_html=True)
                    elif src == "name":
                        st.markdown('<span class="det-badge name">◈ name match</span>',
                                    unsafe_allow_html=True)
                    if new_label != det["label"]:
                        st.session_state["detections"][i]["label"] = new_label
                        if i < len(st.session_state.get("match_sources", [])):
                            st.session_state["match_sources"][i] = None
                        st.rerun()

                c_conf.markdown(f'<div class="det-conf">{det["conf"]:.0%}</div>',
                                unsafe_allow_html=True)

                with c_del:
                    st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                    if st.button("✕", key=f"del_{i}"):
                        to_delete.append(i)
                    st.markdown('</div>', unsafe_allow_html=True)

            if to_delete:
                for idx in sorted(to_delete, reverse=True):
                    st.session_state["detections"].pop(idx)
                    if idx < len(st.session_state.get("match_sources", [])):
                        st.session_state["match_sources"].pop(idx)
                # Reset canvas count so indices stay consistent
                st.session_state["canvas_known_count"] = 0
                st.rerun()

    else:
        st.markdown("""
        <div class="empty">
            <span class="empty-glyph">◈</span>
            <p>Upload an image and click<br>Identify Sea Life to begin.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">AquaScan &nbsp;◼&nbsp; Sea Life Detection &nbsp;◼&nbsp; YOLOv8</div>
""", unsafe_allow_html=True)
