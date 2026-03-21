# -*- coding: utf-8 -*-
"""
Streamlit Evaluation Interface for GPR Buried Object Detection.

Usage:
    streamlit run app.py

Features:
    - Upload a GPR B-scan image
    - Detect and classify buried objects (Utility vs Cavity)
    - Side-by-side visualization with colored bounding boxes
    - Detection results table with confidence scores
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="GPR Buried Object Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }

    .header-container h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .header-container p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }

    .stat-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    .utility-badge {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .cavity-badge {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .detection-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    .upload-section {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: border-color 0.3s ease;
    }

    .upload-section:hover {
        border-color: rgba(102, 126, 234, 0.8);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
CLASS_COLORS = {
    0: (59, 130, 246),   # Blue for Utility (BGR)
    1: (68, 68, 239),    # Red for Cavity (BGR)
}
CLASS_COLORS_RGB = {
    0: (59, 130, 246),   # Blue
    1: (239, 68, 68),    # Red
}
CLASS_NAMES = {0: "Utility", 1: "Cavity"}
CLASS_ICONS = {0: "🔵", 1: "🔴"}


def draw_detections(image, results, conf_threshold=0.25):
    """Draw bounding boxes on the image and return detection data."""
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

                # Colors (BGR for OpenCV)
                if cls_id == 0:
                    color = (246, 130, 59)  # Blue in BGR
                else:
                    color = (68, 68, 239)   # Red in BGR

                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Label background
                label = f"{CLASS_NAMES.get(cls_id, f'cls_{cls_id}')} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                detections.append({
                    "Class": CLASS_NAMES.get(cls_id, f"Unknown ({cls_id})"),
                    "Confidence": f"{conf:.1%}",
                    "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                    "cls_id": cls_id,
                })

    return img, detections


@st.cache_resource
def load_model(model_path):
    """Load and cache the YOLO model."""
    from ultralytics import YOLO
    return YOLO(model_path)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Model path
    default_model = "runs/detect/sanity_check/weights/best.pt"
    model_path = st.text_input(
        "Model path",
        value=default_model,
        help="Path to the trained YOLOv8 .pt weights file",
    )

    # Confidence threshold
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Only show detections above this confidence",
    )

    st.divider()

    st.markdown("### 📋 Class Legend")
    st.markdown(
        '<span class="utility-badge">🔵 Utility</span> '
        "&nbsp; Pipes, cables",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="cavity-badge">🔴 Cavity</span> '
        "&nbsp; Underground voids",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### ℹ️ About")
    st.markdown(
        "This app uses **YOLOv8** to detect hyperbolic "
        "signatures in GPR B-scan images, classifying "
        "them as buried utilities or cavities."
    )


# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────

# Header
st.markdown("""
<div class="header-container">
    <h1>🔍 GPR Buried Object Detector</h1>
    <p>Upload a Ground Penetrating Radar B-scan image to detect and classify buried objects</p>
</div>
""", unsafe_allow_html=True)

# Check if model exists
model_exists = os.path.exists(model_path)

if not model_exists:
    st.warning(
        f"⚠️ Model not found at `{model_path}`. "
        "Please train the model first using `python train.py`, "
        "or update the model path in the sidebar."
    )

# File uploader
uploaded_file = st.file_uploader(
    "Upload a GPR B-scan image",
    type=["jpg", "jpeg", "png"],
    help="Upload a 224×224 GPR scan patch",
)

if uploaded_file is not None:
    # Load image
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_image)

    if model_exists:
        # Load model and run inference
        with st.spinner("🔄 Running detection..."):
            model = load_model(model_path)
            results = model(img_array, conf=conf_threshold, verbose=False)

        # Draw detections
        annotated_img, detections = draw_detections(img_array, results, conf_threshold)

        # ── Summary Stats ──
        n_total = len(detections)
        n_utility = sum(1 for d in detections if d["cls_id"] == 0)
        n_cavity = sum(1 for d in detections if d["cls_id"] == 1)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{n_total}</div>'
                f'<div class="stat-label">Total Detections</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{n_utility}</div>'
                f'<div class="stat-label">🔵 Utilities</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{n_cavity}</div>'
                f'<div class="stat-label">🔴 Cavities</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Side-by-side images ──
        left, right = st.columns(2)

        with left:
            st.markdown("#### 📷 Original Image")
            st.image(pil_image, use_container_width=True)

        with right:
            st.markdown("#### 🎯 Detections")
            st.image(annotated_img, channels="RGB", use_container_width=True)

        # ── Detection Table ──
        if detections:
            st.markdown("#### 📊 Detection Details")

            for det in detections:
                icon = CLASS_ICONS.get(det["cls_id"], "⚪")
                badge_class = "utility-badge" if det["cls_id"] == 0 else "cavity-badge"
                st.markdown(
                    f'<div class="detection-card">'
                    f'<span class="{badge_class}">{icon} {det["Class"]}</span>'
                    f'&nbsp;&nbsp; Confidence: <strong>{det["Confidence"]}</strong>'
                    f'&nbsp;&nbsp; Box: ({det["X1"]}, {det["Y1"]}) → ({det["X2"]}, {det["Y2"]})'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No buried objects detected in this image. Try lowering the confidence threshold.")
    else:
        # No model — just show the uploaded image
        st.image(pil_image, caption="Uploaded image (no model loaded)", use_container_width=True)

else:
    # No image uploaded — show upload prompt
    st.markdown("""
    <div class="upload-section">
        <h3 style="color: rgba(255,255,255,0.7); margin-top: 0;">👆 Upload an image to get started</h3>
        <p style="color: rgba(255,255,255,0.4);">Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
