# -*- coding: utf-8 -*-
"""
Premium Streamlit Evaluation Interface for GPR Buried Object Detection.
"""

import sys
import subprocess
import streamlit as st
import numpy as np
from PIL import Image
import os

# Self-healing OpenCV import for Streamlit Community Cloud's broken Debian dependencies
try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGround | GPR AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Ultra-Premium Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Theme */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Deep Space Gradient Background */
    .stApp {
        background-color: #09090b;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(79, 70, 229, 0.08) 0%, transparent 35%),
            radial-gradient(circle at 85% 30%, rgba(236, 72, 153, 0.06) 0%, transparent 35%),
            radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.04) 0%, transparent 40%);
        background-attachment: fixed;
    }

    /* Frosted Glass Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(9, 9, 11, 0.6) !important;
        backdrop-filter: blur(20px) saturate(150%);
        border-right: 1px solid rgba(255, 255, 255, 0.03);
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.05);
    }

    /* Hero Panel (Header) */
    .hero-panel {
        background: linear-gradient(145deg, rgba(24, 24, 27, 0.6), rgba(9, 9, 11, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.7);
        border-radius: 28px;
        padding: 3.5rem 2rem;
        text-align: center;
        margin-bottom: 3rem;
        backdrop-filter: blur(16px);
        animation: fadeInDown 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .hero-panel h1 {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    .hero-panel p {
        color: #a1a1aa;
        font-size: 1.15rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Metric/Glass Cards */
    .glass-card {
        background: rgba(24, 24, 27, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 20px;
        padding: 1.75rem;
        backdrop-filter: blur(12px);
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        opacity: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.1);
        transform: translateY(-5px);
        box-shadow: 0 20px 40px -8px rgba(0, 0, 0, 0.4);
        background: rgba(39, 39, 42, 0.5);
    }
    .stat-number {
        font-size: 3rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }
    .stat-number.blue { background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stat-number.red { background: linear-gradient(to right, #fb7185, #f43f5e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }

    /* Elegant Detection Item */
    .detection-row {
        background: rgba(24, 24, 27, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        opacity: 0;
    }
    .detection-row:hover {
        background: rgba(39, 39, 42, 0.6);
        border-color: rgba(255, 255, 255, 0.08);
    }
    
    .det-left { display: flex; align-items: center; gap: 1rem; }
    .det-right { text-align: right; }
    .det-coords { font-size: 0.8rem; color: #71717a; font-family: monospace; background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 6px; }

    /* Badges */
    .badge {
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-utility {
        background: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    .badge-cavity {
        background: rgba(244, 63, 94, 0.1);
        color: #f43f5e;
        border: 1px solid rgba(244, 63, 94, 0.2);
    }

    /* Images Panels */
    .image-panel {
        background: rgba(9, 9, 11, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.04);
        padding: 1rem;
        border-radius: 24px;
        box-shadow: inset 0 2px 20px rgba(0,0,0,0.5);
    }

    /* Custom Uploader Dropzone styling */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(24, 24, 27, 0.3) !important;
        border: 2px dashed rgba(255, 255, 255, 0.08) !important;
        border-radius: 24px !important;
        padding: 3rem !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #818cf8 !important;
        background-color: rgba(129, 140, 248, 0.03) !important;
    }
    .upload-hint {
        text-align: center;
        margin-top: -15px;
        color: #71717a;
        font-size: 0.9rem;
    }

    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .d-1 { animation-delay: 0.1s; }
    .d-2 { animation-delay: 0.2s; }
    .d-3 { animation-delay: 0.3s; }
    .d-4 { animation-delay: 0.4s; }
    .d-5 { animation-delay: 0.5s; }
    
    /* Headers inside markdown */
    h3, h4 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        margin-bottom: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Architecture / Helpers
# ──────────────────────────────────────────────
# Vibrant BGR colors for OpenCV drawing
COLOR_UTILITY = (255, 187, 51)  # Vivid Skublue/Cyan (OpenCV uses BGR -> #33bbff)
COLOR_CAVITY  = (81, 63, 244)   # Vivid Rose/Pink (OpenCV uses BGR -> #f43f51)

CLASS_NAMES = {0: "Utility", 1: "Cavity"}

def draw_premium_detections(image, results, conf_threshold=0.25):
    """Draw highly-styled bounding boxes on the image and return detection metrics."""
    img = image.copy()
    detections = []

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue

                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Assign colors based on class
                color = COLOR_UTILITY if cls_id == 0 else COLOR_CAVITY
                
                # Draw thick, beautiful bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                # Draw solid background for label
                label = f"{CLASS_NAMES.get(cls_id, 'Anomaly')} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
                
                # Dark transparent backing for text
                overlay = img.copy()
                cv2.rectangle(overlay, (x1, y1 - th - 12), (x1 + tw + 10, y1), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                
                # Vibrant text border for legibility
                cv2.putText(img, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1)

                detections.append({
                    "Class": CLASS_NAMES.get(cls_id, "Unknown"),
                    "Confidence": conf,
                    "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                    "cls_id": cls_id,
                })

    # Sort detections by confidence (highest first)
    detections.sort(key=lambda x: x["Confidence"], reverse=True)
    return img, detections


@st.cache_resource(show_spinner=False)
def load_model(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)


# ──────────────────────────────────────────────
# Sidebar Structure
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='font-weight:800; color:white; letter-spacing:-1px; margin-bottom: 2rem;'>DeepGround</h2>", unsafe_allow_html=True)
    
    st.markdown("<p style='color:#a1a1aa; font-weight:600; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;'>Model Configuration</p>", unsafe_allow_html=True)
    
    # Sleek Model Path Input
    default_model = "models/best_model.pt"
    model_path = st.text_input(
        "Weights Path",
        value=default_model,
    )

    # Sleek Threshold Slider
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
    )

    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

    # Legend Display
    st.markdown("<p style='color:#a1a1aa; font-weight:600; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;'>Detection Legend</p>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style='display: flex; flex-direction: column; gap: 1rem; margin-top: 1rem;'>
            <div style='display: flex; align-items: center; gap: 1rem;'>
                <div class='badge badge-utility'>Utility</div>
                <span style='color:#d4d4d8; font-size:0.9rem;'>Pipes, Cables</span>
            </div>
            <div style='display: flex; align-items: center; gap: 1rem;'>
                <div class='badge badge-cavity'>Cavity</div>
                <span style='color:#d4d4d8; font-size:0.9rem;'>Subsurface Voids</span>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#52525b; font-size:0.8rem; text-align:center;'>Powered by YOLOv8 Vision Architecture</p>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main Dashboard
# ──────────────────────────────────────────────

# Massive Premium Hero Banner
st.markdown("""
<div class="hero-panel">
    <h1>Subsurface Anomaly Intelligence</h1>
    <p>Upload Ground Penetrating Radar B-scans to instantly identify and classify buried anomalies using advanced computer vision.</p>
</div>
""", unsafe_allow_html=True)

model_exists = os.path.exists(model_path)
if not model_exists:
    st.error(f"⚠️ **Model file not found** at `{model_path}`. Ensure you have run the training script.")

# Premium Uploader
st.markdown("<h3 style='color: white; margin-bottom: 0.5rem;'>Input Telemetry</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.markdown("<p class='upload-hint'>Drag and drop a 224×224 B-scan patch to begin analysis.</p>", unsafe_allow_html=True)

if uploaded_file is not None and model_exists:
    # Processing
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_image)

    # Inference without breaking the UI flow
    model = load_model(model_path)
    results = model(img_array, conf=conf_threshold, verbose=False)

    annotated_img, detections = draw_premium_detections(img_array, results, conf_threshold)

    st.markdown("<hr style='margin: 3rem 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)

    # ── METRIC CARDS ──
    st.markdown("<h3 style='color: white;'>Analysis Overview</h3>", unsafe_allow_html=True)
    
    n_total = len(detections)
    n_util = sum(1 for d in detections if d["cls_id"] == 0)
    n_cav = sum(1 for d in detections if d["cls_id"] == 1)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="glass-card d-1">
            <div class="stat-number">{n_total}</div>
            <div class="stat-label">Total Signatures</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="glass-card d-2">
            <div class="stat-number blue">{n_util}</div>
            <div class="stat-label">Verified Utilities</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="glass-card d-3">
            <div class="stat-number red">{n_cav}</div>
            <div class="stat-label">Verified Cavities</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── VISUALIZATION PANELS ──
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("<h4 style='color: #a1a1aa;'>Raw Feed</h4>", unsafe_allow_html=True)
        st.markdown("<div class='image-panel d-4'>", unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_right:
        st.markdown("<h4 style='color: white;'>Machine Vision Out</h4>", unsafe_allow_html=True)
        st.markdown("<div class='image-panel d-5'>", unsafe_allow_html=True)
        st.image(annotated_img, channels="RGB", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── DETECTION LOG ──
    if detections:
        st.markdown("<h3 style='color: white;'>Signature Log</h3>", unsafe_allow_html=True)
        
        # Calculate dynamic delay for list items
        for i, det in enumerate(detections):
            badge_type = "badge-utility" if det["cls_id"] == 0 else "badge-cavity"
            conf_pct = det["Confidence"] * 100
            
            # Confidence bar length
            bar_color = "#38bdf8" if det["cls_id"] == 0 else "#f43f5e"
            
            st.markdown(f"""
            <div class="detection-row" style="animation-delay: {0.1 + (i*0.05)}s;">
                <div class="det-left">
                    <div class="badge {badge_type}">{det["Class"]}</div>
                    <div style="width: 120px; height: 6px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;">
                        <div style="width: {conf_pct}%; height: 100%; background: {bar_color};"></div>
                    </div>
                    <span style="font-weight: 700; color: white;">{conf_pct:.1f}%</span>
                </div>
                <div class="det-right">
                    <span class="det-coords">({det['X1']}, {det['Y1']}) ↗ ({det['X2']}, {det['Y2']})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem; background: rgba(24,24,27,0.3); border-radius: 20px; border: 1px dashed rgba(255,255,255,0.1);">
            <h4 style="color:#71717a; margin:0;">Zero signatures detected at current confidence threshold.</h4>
        </div>
        """, unsafe_allow_html=True)
