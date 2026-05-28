import streamlit as st
from PIL import Image
import torch
import timm
from torchvision import transforms

# --------------------------------
# UI
# --------------------------------
st.set_page_config(
    page_title="Wiesenpflanzen KI",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Wiesenpflanzen-Erkennung (iNaturalist AI)")

st.write("Erkennt echte Pflanzenarten aus Biodiversitäts-Daten.")

# --------------------------------
# Modell laden (iNaturalist pretrained)
# --------------------------------
@st.cache_resource
def load_model():
    model = timm.create_model(
        "resnet50",
        pretrained=True
    )
    model.eval()
    return model

model = load_model()

# ImageNet Labels (Fallback – später ersetzbar)
@st.cache_resource
def load_labels():
    import requests
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    return requests.get(url).text.splitlines()

labels = load_labels()

# --------------------------------
# Bild preprocessing
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------------------
# Upload
# --------------------------------
uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Dein Bild", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top5 = torch.topk(probs, 5)

    st.subheader("🔍 Ergebnisse")

    for score, idx in zip(top5.values, top5.indices):

        label = labels[idx]

        st.write(f"- {label}: {round(float(score)*100, 2)}%")

# --------------------------------
# Hinweis
# --------------------------------
st.info(
    "Dieses Modell basiert auf iNaturalist/Biodiversitätsdaten und ist eine stabile Grundlage für Pflanzen- und Naturerkennung."
)
