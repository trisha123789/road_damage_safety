
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Road Damage Detection AI",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(to right,#0f172a,#111827);
        color:white;
    }

    .title {
        text-align:center;
        font-size:60px;
        font-weight:bold;
        color:#38bdf8;
    }

    .subtitle {
        text-align:center;
        font-size:22px;
        color:#cbd5e1;
        margin-bottom:30px;
    }

    .card {
        background:#1e293b;
        padding:20px;
        border-radius:20px;
        box-shadow:0px 0px 20px rgba(0,0,0,0.5);
    }

    .prediction {
        text-align:center;
        font-size:40px;
        font-weight:bold;
        color:#22c55e;
    }

    .confidence {
        text-align:center;
        font-size:24px;
        color:#facc15;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🚧 Road Damage Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Smart City AI Monitoring using CNN + ONNX Runtime</div>',
    unsafe_allow_html=True
)

# =========================================================
# LOAD ONNX MODEL
# =========================================================

@st.cache_resource

def load_model():
    session = ort.InferenceSession("road_damage_model.onnx")
    return session

session = load_model()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# =========================================================
# CLASS LABELS
# =========================================================

classes = [
    "crack",
    "manhole",
    "pothole"
]

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ AI Dashboard")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Upload road images to detect:

    ✅ Cracks
    ✅ Potholes
    ✅ Manholes

    CNN inference powered by ONNX Runtime.
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Features")

st.sidebar.write("✔ Real-Time Prediction")
st.sidebar.write("✔ Confidence Score")
st.sidebar.write("✔ Prediction Probabilities")
st.sidebar.write("✔ Confusion Matrix Visualization")
st.sidebar.write("✔ Beautiful Analytics UI")
st.sidebar.write("✔ TensorFlow-Free Deployment")

# =========================================================
# IMAGE PREPROCESSING
# =========================================================

IMG_SIZE = 64


def preprocess_image(image):

    image = np.array(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype(np.float32) / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# MAIN PREDICTION UI
# =========================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        with st.spinner("Analyzing Road Condition..."):

            time.sleep(2)

            processed = preprocess_image(image)

            prediction = session.run(
                [output_name],
                {input_name: processed}
            )[0]

            predicted_index = int(np.argmax(prediction))

            confidence = float(np.max(prediction)) * 100

            predicted_label = classes[predicted_index]

        st.success("Analysis Completed")

        st.markdown(
            f'<div class="prediction">{predicted_label.upper()}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">Confidence: {confidence:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.progress(confidence / 100)

        # =====================================================
        # ALERT SYSTEM
        # =====================================================

        if predicted_label == "pothole":
            st.error("⚠ Severe Road Damage Detected")

        elif predicted_label == "crack":
            st.warning("⚠ Road Crack Detected")

        elif predicted_label == "manhole":
            st.info("ℹ Manhole Detected")

        # =====================================================
        # PROBABILITIES
        # =====================================================

        st.subheader("📈 Prediction Probabilities")

        probs = prediction[0]

        prob_df = pd.DataFrame({
            "Class": classes,
            "Probability": probs
        })

        st.bar_chart(
            prob_df.set_index("Class")
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # ANALYTICS SECTION
    # =========================================================

    st.markdown("---")

    st.subheader("📊 AI Analytics Dashboard")

    tab1, tab2, tab3 = st.tabs([
        "Prediction Insights",
        "Confusion Matrix",
        "System Info"
    ])

    with tab1:

        st.markdown("### CNN Prediction Summary")

        st.write(f"Predicted Class: **{predicted_label}**")
        st.write(f"Confidence Score: **{confidence:.2f}%**")

        if confidence > 90:
            st.success("High Confidence Prediction")

        elif confidence > 70:
            st.warning("Moderate Confidence Prediction")

        else:
            st.error("Low Confidence Prediction")

    with tab2:

        st.markdown("### Sample Confusion Matrix")

        sample_cm = np.array([
            [45, 2, 1],
            [3, 40, 4],
            [2, 5, 50]
        ])

        fig, ax = plt.subplots(figsize=(6,5))

        sns.heatmap(
            sample_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            ax=ax
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        st.pyplot(fig)

    with tab3:

        st.markdown("### System Information")

        st.write("Model Type: CNN")
        st.write("Deployment Runtime: ONNX Runtime")
        st.write("Input Size: 64x64")
        st.write("Framework: Streamlit")
        st.write("Deployment Ready: Yes")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <h4 style='color:gray;'>
    Smart City Road Intelligence System 🚀<br>
    Deep Learning + Streamlit + ONNX Runtime
    </h4>
    </center>
    """,
    unsafe_allow_html=True
)



