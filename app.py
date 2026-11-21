import io
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.set_page_config(page_title="AIML-501 Pneumonia Model Demo", page_icon="🩻", layout="wide")
st.title("Chest X-Ray Pneumonia Classification")
st.caption("AIML-501 | Model Development Residency | Educational demonstration")
st.warning("Educational use only. Not intended for clinical diagnosis, triage, or treatment. Do not upload identifiable patient images.")

@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    device = torch.device("cpu")
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    densenet = models.densenet121(weights=None)
    densenet.classifier = nn.Linear(densenet.classifier.in_features, 2)
    for model, filename in [
        (resnet, "resnet18_pneumonia.pth"),
        (densenet, "densenet121_pneumonia.pth"),
    ]:
        path = BASE_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing model weights: {filename}")
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
    return {"ResNet18": resnet, "DenseNet121": densenet}

@torch.inference_mode()
def predict(model, image):
    tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    index = int(max(range(len(probabilities)), key=lambda i: probabilities[i]))
    return CLASSES[index], probabilities

try:
    loaded_models = load_models()
except Exception as exc:
    st.error(f"Could not load the saved models: {exc}")
    st.stop()

st.subheader("Upload an example chest X-ray")
file = st.file_uploader("Choose a PNG or JPG image", type=["png", "jpg", "jpeg"])
if file is not None:
    try:
        image = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file could not be read as an image.")
        st.stop()
    left, right = st.columns([1, 2])
    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with right:
        rows = []
        for model_name, model in loaded_models.items():
            label, probs = predict(model, image)
            rows.append({
                "Model": model_name,
                "Predicted class": label,
                "NORMAL probability": probs[0],
                "PNEUMONIA probability": probs[1],
            })
        st.subheader("Model predictions")
        st.dataframe(pd.DataFrame(rows).style.format({
            "NORMAL probability": "{:.1%}",
            "PNEUMONIA probability": "{:.1%}",
        }), use_container_width=True, hide_index=True)
        st.caption("These are model output probabilities, not clinically validated disease probabilities.")
else:
    st.info("Upload a non-identifiable educational example to compare predictions.")

with st.expander("Day 1 evaluation results (if available)"):
    csv_path = BASE_DIR / "model_comparison_results.csv"
    if csv_path.is_file():
        try:
            st.dataframe(pd.read_csv(csv_path), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Could not display the results file: {exc}")
    else:
        st.write("The optional comparison CSV was not included.")

st.divider()
st.caption("AIML-501: Model Development | Demo only — not a clinical decision-support system")
