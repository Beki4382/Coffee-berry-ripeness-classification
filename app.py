import io

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(page_title="Coffee Berry Ripeness", layout="centered")
st.title("Coffee Berry Ripeness Classification")
st.write("Upload an image to classify coffee berry ripeness.")


@st.cache_resource
def load_model():
    return YOLO("best.pt")


def predict(image: Image.Image):
    model = load_model()
    img_array = np.array(image.convert("RGB"))
    results = model.predict(source=img_array, verbose=False)
    if not results:
        return None, None

    result = results[0]
    if getattr(result, "probs", None) is not None:
        probs = result.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        label = result.names.get(top1_idx, str(top1_idx))
        return label, top1_conf

    # Fallback for detection-style outputs
    if getattr(result, "boxes", None) is not None and result.boxes:
        labels = []
        for cls_idx, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            label = result.names.get(int(cls_idx), str(int(cls_idx)))
            labels.append((label, float(conf)))
        return labels, None

    return None, None


uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read()))
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run classification"):
        with st.spinner("Running model..."):
            label, confidence = predict(image)

        if label is None:
            st.error("No prediction returned.")
        elif isinstance(label, list):
            st.subheader("Detections")
            for det_label, det_conf in label:
                st.write(f"{det_label}: {det_conf:.2%}")
        else:
            st.subheader("Prediction")
            st.write(f"Label: {label}")
            st.write(f"Confidence: {confidence:.2%}")
