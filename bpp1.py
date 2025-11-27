import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import torch


@st.cache_resource
def load_model():
    return YOLO("yolo11n-resnet.pt")   

model = load_model()


DATASET_PATH = "dataset/waste_classifcation"   
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR   = os.path.join(DATASET_PATH, "val")
TEST_DIR  = os.path.join(DATASET_PATH, "test")

def get_class_folders(path):
    if os.path.exists(path):
        return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return []


st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="centered")

st.title("♻️ Biodegradable vs Non-Biodegradable Classifier")
st.write("Upload an image and the model will classify it.")

with st.expander("📂 View Dataset Class Information"):
    train_classes = get_class_folders(TRAIN_DIR)

    if train_classes:
        st.subheader("Dataset Classes:")
        st.write(train_classes)
    else:
        st.write("Dataset not found. Place it inside a folder named `dataset/`.")


uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:


    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)


    results = model(img)

    pred_idx  = int(results[0].probs.top1)
    pred_name = results[0].names[pred_idx]
    confidence = float(results[0].probs.top1conf) * 100


    if pred_name.lower() in ["bio", "biodegradable"]:
        final = "Biodegradable"
        color = "green"
    else:
        final = "Non-Biodegradable"
        color = "red"


    st.markdown(f"""
        <div style="
            padding: 25px;
            margin-top: 25px;
            border-radius: 18px;
            background: linear-gradient(135deg, #d9f7ff, #f1f0ff);
            text-align:center;
            box-shadow: 0 8px 22px rgba(0,0,0,0.25);
        ">
            <h2 style="color:{color}; font-size:40px; margin:0;">
                {final}
            </h2>
            <p style="font-size:18px; opacity:0.7;">
                Confidence: {confidence:.2f}%
            </p>
        </div>
    """, unsafe_allow_html=True)
