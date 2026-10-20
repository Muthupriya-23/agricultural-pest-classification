
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st

from PIL import Image
from skimage.feature import hog, local_binary_pattern

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Agricultural Pest Detection",
    page_icon="🐛",
    layout="wide"
)

MODEL_PATH = Path("models/pest_classifier.joblib")
CLASSES_PATH = Path("models/classes.joblib")
IMAGE_SIZE = (128, 128)

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    classes = joblib.load(CLASSES_PATH)
    return model, classes


# -------------------------------
# Image feature extraction
# Must match train_model.py
# -------------------------------
def extract_features(image):
    image = cv2.resize(image, IMAGE_SIZE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    lbp = local_binary_pattern(
        gray, 8, 1, method="uniform"
    )

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10)
    )

    lbp_hist = lbp_hist.astype("float32")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    ).flatten()

    color_hist /= (color_hist.sum() + 1e-7)

    return np.concatenate([
        hog_features,
        lbp_hist,
        color_hist
    ]).reshape(1, -1)


# -------------------------------
# Website interface
# -------------------------------
st.title("🐛 Agricultural Pest Classification")
st.write(
    "Upload an agricultural pest image to obtain "
    "a machine-learning-based category prediction."
)

st.info(
    "This is an educational prototype. Predictions may be "
    "incorrect, especially because the current model's test "
    "accuracy is approximately 30.85%."
)

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error(
        "Trained model files not found. "
        "Run `python train_model.py` first."
    )
    st.stop()

model, classes = load_model()

st.sidebar.header("Project Information")
st.sidebar.write("**Project:** Agricultural Pest Classification")
st.sidebar.write("**Algorithm:** Support Vector Machine (SVM)")
st.sidebar.write("**Feature extraction:** HOG, LBP and colour histogram")
st.sidebar.write("**Categories:**", len(classes))
st.sidebar.write("**Test accuracy:** 30.85%")

uploaded_file = st.file_uploader(
    "Choose a pest image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(pil_image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(pil_image, use_container_width=True)

    with col2:
        st.subheader("Classification Result")

        if st.button("Classify Pest", type="primary"):
            with st.spinner("Analysing image..."):
                # Convert RGB image to OpenCV BGR format
                bgr_image = cv2.cvtColor(
                    image_array, cv2.COLOR_RGB2BGR
                )

                features = extract_features(bgr_image)
                prediction = model.predict(features)[0]

                st.success(f"Predicted category: {prediction}")

                st.warning(
                    "This is a model prediction, not a confirmed "
                    "identification. Please verify the result."
                )

                st.caption(
                    "The current model does not provide a reliable "
                    "probability of correctness."
                )

st.divider()

st.subheader("Project Modules")
st.markdown("""
- **Module 1:** Pest image input and preprocessing
- **Module 2:** Image enhancement and feature extraction
- **Module 3:** Pest classification and result display
""")

st.caption("Agricultural Pest Classification | Student Project")
