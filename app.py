import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Brain Tumor Diagnostic Suite",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for high-contrast text and specific Red Headings
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }

    /* Metric boxes: White background, Black text */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Forcing black text on labels and values */
    div[data-testid="stMetricLabel"] { color: #000000 !important; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #1E3A8A !important; }

    h1, h2, h3 { color: #1E3A8A; }
    .stMarkdown { color: #000000; }

    /* Custom Red Headers for the Images */
    .red-header {
        color: #FF0000 !important;
        font-weight: bold;
        font-size: 1.8rem; /* Increased size */
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODELS ---
cls_model_path = "cls_model.pt"
seg_model_path = "seg_model.pt"

@st.cache_resource
def load_models(c_path, s_path):
    cls_model = YOLO(c_path) if os.path.exists(c_path) else None
    seg_model = YOLO(s_path) if os.path.exists(s_path) else None
    return cls_model, seg_model


cls_net, seg_net = load_models(cls_model_path, seg_model_path)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2491/2491413.png", width=80)
    st.title("Control Panel")
    st.markdown("### Model Settings")
    # Increased default confidence to 0.50 (50%)
    conf_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.50, 0.05)
    st.divider()
    # Removed "Models Ready" message as requested. Only show error if missing.
    if not cls_net or not seg_net:
        st.error("⚠️ Check Model Paths")

# --- 4. MAIN INTERFACE ---
st.title("🧠 Brain Tumor Diagnostic Assistant")
st.write("Professional AI-assisted analysis for Glioma, Meningioma, and Pituitary tumors.")
st.markdown("---")

if not cls_net:
    st.error(f"Classification Model not found.")
else:
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        bytes_data = uploaded_file.read()
        image_array = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(image_array, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # STEP 1: Classification (The "Doctor")
        cls_results = cls_net.predict(source=img_rgb, imgsz=224)
        top_idx = cls_results[0].probs.top1
        label = cls_results[0].names[top_idx].upper()
        confidence = float(cls_results[0].probs.top1conf)

        # STEP 2: Segmentation (The "Radiologist")
        seg_results = seg_net.predict(source=img_rgb, imgsz=320, conf=conf_threshold) if seg_net else None

        # --- DISPLAY RESULTS ---
        # Adjusted ratios [1.2, 1.2, 0.6] to make images larger
        col1, col2, col3 = st.columns([1.2, 1.2, 0.6])

        with col1:
            st.markdown('<p class="red-header">📷 Input MRI</p>', unsafe_allow_html=True)
            st.image(img_rgb, use_container_width=True)

        with col2:
            st.markdown('<p class="red-header">🎭 Tumor Mask & Box</p>', unsafe_allow_html=True)

            # Safety Filter: Only show boxes if Classifier says it's NOT healthy
            if label != "NOTUMOR" and seg_results and seg_results[0].masks is not None:
                res_plotted = seg_results[0].plot(labels=False, boxes=True, masks=True)
                st.image(res_plotted, use_container_width=True)
                st.caption(f"Visualized Shape for: {label}")
            else:
                # Show clean image if Healthy or no mask found
                st.image(img_rgb, use_container_width=True)
                st.caption("No clinical anomalies visualized.")

        with col3:
            st.markdown("### 📊 Diagnosis")
            if label == "NOTUMOR":
                st.metric("Status", "HEALTHY", delta="Clear", delta_color="normal")
                st.success("No anomalies detected.")
            else:
                st.metric("Detected Class", label, delta=f"{confidence:.1%}", delta_color="inverse")
                st.error(f"Attention: {label} detected.")

            with st.expander("Diagnostic Data"):
                st.markdown(f"**Final Diagnosis:** `{label}`")
                st.markdown(f"**Confidence:** `{confidence:.4f}`")
                if label != "NOTUMOR" and seg_results and seg_results[0].masks is not None:
                    num_m = len(seg_results[0].masks)
                    st.markdown(f"**Detected Shapes:** `{num_m}`")
                else:
                    st.markdown("**Detected Shapes:** `0`")

        st.divider()
        st.subheader("Classification Probability Breakdown")
        probs = cls_results[0].probs.data.tolist()
        names = cls_results[0].names
        chart_data = {names[i]: probs[i] for i in range(len(probs))}
        st.bar_chart(chart_data)

    else:
        st.info("Upload an image to generate the diagnostic report.")

st.markdown("""
    <div style='text-align: center; color: grey; padding: 20px; border-top: 1px solid #eee;'>
        <small>AI Diagnostic Assistant | Dual YOLOv11 Pipeline | Professional Edition</small>
    </div>
""", unsafe_allow_html=True)