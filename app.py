
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import random

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }

    h1, h2, h3 {
        color: #38bdf8;
    }

    .stButton>button {
        background: linear-gradient(90deg,#2563eb,#06b6d4);
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        border: none;
    }

    .prediction-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #38bdf8;
    }

    .recommendation-box {
        background-color: #111827;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #22c55e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# TITLE
# ======================================================

st.title("🛣️ AI-Based Road Damage Detection System")
st.subheader("Smart City Infrastructure Monitoring using CNN")

# ======================================================
# SECTION 2 — ABOUT PROJECT
# ======================================================

st.header("📘 About the Project")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        ### Why Road Monitoring is Important

        - Prevents accidents
        - Improves transportation safety
        - Reduces vehicle damage
        - Supports smart city infrastructure
        - Helps governments prioritize repairs
        """
    )

with col2:
    st.success(
        """
        ### Role of CNN in Computer Vision

        - Detects cracks and potholes
        - Learns image patterns automatically
        - Provides high prediction accuracy
        - Processes real-time road images
        - Used widely in autonomous systems
        """
    )

with col3:
    st.warning(
        """
        ### Practical Industry Applications

        - Smart city surveillance
        - Highway monitoring systems
        - Autonomous vehicles
        - Municipal maintenance systems
        - AI-powered inspection drones
        """
    )

# ======================================================
# BUILT-IN SAMPLE DATASET
# ======================================================



# ======================================================
# SECTION 3 — UPLOAD AREA
# ======================================================

st.header("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

# ======================================================
# LOAD IMAGE
# ======================================================

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
else:
    try:
        image = Image.open(sample_images[selected_sample])
    except:
        st.warning("Sample image not found.")

# ======================================================
# SECTION 4 — IMAGE PREVIEW
# ======================================================

if image is not None:
    st.header("🖼️ Uploaded Image Preview")

    st.image(image, caption="Road Image", use_container_width=True)

# ======================================================
# DUMMY CNN PREDICTION FUNCTION
# ======================================================

classes = [
    "Pothole",
    "Crack",
    "Normal Road",
    "Road Patch"
]

severity_map = {
    "Pothole": "High",
    "Crack": "Medium",
    "Normal Road": "Low",
    "Road Patch": "Low"
}

recommendations = {
    "Pothole": "Immediate maintenance recommended. High-risk road condition detected.",
    "Crack": "Schedule repair soon to prevent further damage.",
    "Normal Road": "Road condition appears safe.",
    "Road Patch": "Monitor patched region regularly for future deterioration."
}


def predict_damage():

    probabilities = np.random.dirichlet(np.ones(len(classes)), size=1)[0]

    predicted_index = np.argmax(probabilities)

    prediction = classes[predicted_index]

    confidence = probabilities[predicted_index] * 100

    return prediction, confidence, probabilities

# ======================================================
# SECTION 5 — PREDICTION AREA
# ======================================================

if image is not None:

    st.header("🤖 Prediction Area")

    if st.button("Analyze Road Damage"):

        prediction, confidence, probabilities = predict_damage()

        severity = severity_map[prediction]

        st.markdown(
            f"""
            <div class='prediction-box'>
            <h2>Prediction: {prediction} Detected</h2>
            <h3>Confidence: {confidence:.2f}%</h3>
            <h3>Severity: {severity}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ======================================================
        # SECTION 6 — VISUALIZATION AREA
        # ======================================================

        st.header("📊 Visualization Area")

        chart_data = pd.DataFrame({
            "Damage Type": classes,
            "Confidence": probabilities * 100
        })

        st.subheader("Class Confidence Graph")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            chart_data["Damage Type"],
            chart_data["Confidence"]
        )

        ax.set_ylabel("Confidence (%)")
        ax.set_xlabel("Classes")
        ax.set_title("Road Damage Prediction Confidence")

        st.pyplot(fig)

        st.subheader("Probability Distribution")

        fig2, ax2 = plt.subplots(figsize=(7, 7))

        ax2.pie(
            probabilities,
            labels=classes,
            autopct='%1.1f%%'
        )

        st.pyplot(fig2)

        # ======================================================
        # SECTION 7 — RECOMMENDATIONS
        # ======================================================

        st.header("🛠️ Recommendations")

        st.markdown(
            f"""
            <div class='recommendation-box'>
            <h3>{recommendations[prediction]}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ======================================================
        # EXTRA FEATURES
        # ======================================================

        st.header("📌 Additional Insights")

        if severity == "High":
            st.error("⚠️ Severe road damage detected. Immediate authority action required.")

        elif severity == "Medium":
            st.warning("⚠️ Moderate damage detected. Maintenance recommended.")

        else:
            st.success("✅ Road condition is relatively stable.")

        st.metric(
            label="AI Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown(
    """
    ### 🚀 Smart City Vision

    AI-based road monitoring systems help governments automate infrastructure inspection,
    reduce manual labor, improve public safety, and support future smart transportation systems.
    """
)



