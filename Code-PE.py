import streamlit as st
from PIL import Image
import torch
import timm
from torchvision import transforms

# --------------------------------
# UI
# --------------------------------
st.set_page_config(
    page_title="Wiesenpflanzen KI - Stabil",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Wiesenpflanzen KI (Stabile Stufe 3)")
st.write("Robuste lokale Pflanzen-/Naturerkennung")

# --------------------------------
# Giftigkeits-Datenbank
# --------------------------------
TOXICITY_DB = {
    "Urtica": "⚠️ Brennnessel (reizend, essbar nach Zubereitung)",
    "Taraxacum": "✅ Löwenzahn (essbar)",
    "Digitalis": "☠️ Giftig (Fingerhut)",
    "Heracleum": "☠️ Giftig (Bärenklau)"
}

# --------------------------------
# Modell laden (STABIL!)
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

# --------------------------------
# Labels (ImageNet fallback stabil)
# --------------------------------
@st.cache_resource
def load_labels():
    import requests
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    return requests.get(url).text.splitlines()

labels = load_labels()

# --------------------------------
# Image transform
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------------------
# Upload
# --------------------------------
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top5 = torch.topk(probs, 5)

    st.subheader("🔍 Ergebnisse")

    for score, idx in zip(top5.values, top5.indices):

        label = labels[idx]
        confidence = round(float(score) * 100, 2)

        st.write(f"**{label}** — {confidence}%")

        for key, value in TOXICITY_DB.items():
            if key.lower() in label.lower():
                st.warning(value)

# --------------------------------
# Hinweis
# --------------------------------
st.info("Stabile lokale KI-Version (kein API, keine Accounts).")
