# ==========================================
# Chest X-ray Lung Opacity Detection App
# ==========================================

from pathlib import Path

import numpy as np
import pydicom
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


# ==========================================
# 01. Configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "04_Exports"
    / "Models"
    / "resnet18_lung_opacity.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "No Lung Opacity Detected",
    "Lung Opacity Detected"
]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 02. Image Preprocessing
# ==========================================

def preprocess_image(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".dcm"):
        dicom = pydicom.dcmread(uploaded_file)
        image = dicom.pixel_array.astype(np.float32)

        if dicom.get("PhotometricInterpretation") == "MONOCHROME1":
            image = image.max() - image

        image = image - image.min()
        image = image / (image.max() + 1e-8)
        image = (image * 255).astype(np.uint8)

        image = Image.fromarray(image).convert("RGB")

    else:
        image = Image.open(uploaded_file).convert("RGB")

    tensor = TRANSFORM(image).unsqueeze(0)

    return image, tensor


# ==========================================
# 03. Load Trained Model
# ==========================================

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model


# ==========================================
# 04. Application Interface
# ==========================================

st.set_page_config(
    page_title="Chest X-ray Analysis",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Chest X-ray Analysis")

st.write(
    "Upload a chest X-ray to analyze whether the trained "
    "ResNet18 model detects signs of lung opacity."
)

st.warning(
    "Educational research prototype only. "
    "This application does not diagnose pneumonia "
    "and must not be used for clinical decisions."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["dcm", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    try:
        image, input_tensor = preprocess_image(uploaded_file)

        st.image(
            image,
            caption="Uploaded chest X-ray",
            use_container_width=True
        )

        if not uploaded_file.name.lower().endswith(".dcm"):
            st.info(
                "PNG/JPEG images may have different intensity "
                "characteristics from the DICOM images used "
                "during training. Interpret results cautiously."
            )

        if not MODEL_PATH.exists():
            st.error("Trained model checkpoint not found.")

        elif st.button("Analyze X-ray", type="primary"):

            with st.spinner("Analyzing X-ray..."):
                model = load_model()

                with torch.no_grad():
                    outputs = model(input_tensor.to(DEVICE))
                    probabilities = torch.softmax(outputs, dim=1)[0]

                    predicted_class = int(
                        torch.argmax(probabilities).item()
                    )

                    score = float(
                        probabilities[predicted_class].item()
                    )

            st.divider()
            st.subheader("Analysis Result")

            if predicted_class == 1:
                st.warning("Lung Opacity Detected")
            else:
                st.success("No Lung Opacity Detected")

            st.metric(
                "Model confidence score",
                f"{score * 100:.1f}%"
            )

            st.progress(score)

            st.caption(
                "This score is an uncalibrated model output, "
                "not a validated probability of disease. "
                "A negative result does not rule out pneumonia."
            )

    except Exception as error:
        st.error(f"Unable to process image: {error}")