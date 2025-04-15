import streamlit as st
from PIL import Image
from vit_utils import load_model, predict_disease

st.set_page_config(page_title="Lung Disease Detection", layout="centered")

# Load models
@st.cache_resource
def load_models():
    stage1 = load_model("../Model/best_model_stage1.pth", num_labels=4)
    stage2 = load_model("../Model/best_model_stage2.pth", num_labels=2)
    return stage1, stage2

stage1_model, stage2_model = load_models()

# UI
st.title(" Lung Disease Detection using ViT")
st.markdown("Upload a chest X-ray image. The model will predict the condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            result = predict_disease(stage1_model, stage2_model, image)
        st.success(f"Prediction: **{result}**")
