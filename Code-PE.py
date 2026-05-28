import streamlit as st
from PIL import Image
import torch
import timm
from torchvision import transforms

# --------------------------------
# UI
# --------------------------------
st.set_page_config(
    page_title="Wiesenpflanzen KI - Stufe 3",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Wiesenpflanzen KI (iNaturalist Stufe 3)")
st.write("Echte Pflanzen-/Naturarten-Erkennung basierend auf Biodiversitätsdaten")

# --------------------------------
# Giftigkeits-Datenbank (Startversion)
# --------------------------------
TOXICITY_DB = {
    "Urtica dioica": "⚠️ Brennnessel (leicht reizend, aber essbar nach Zubereitung)",
    "Taraxacum officinale": "✅ Löwenzahn (essbar)",
    "Digitalis purpurea": "☠️ Stark giftig (Fingerhut)",
    "Heracleum": "☠️ Giftig (Bärenklau-Gattung)"
}

# --------------------------------
# Modell laden (iNaturalist pretrained)
# --------------------------------
@st.cache_resource
def load_model():
    model = timm.create_model(
        "vit_base_patch16_224.in21k_ft_inat21",
        pretrained=True
    )
    model.eval()
    return model

model = load_model()

# --------------------------------
# Labels laden (iNaturalist Klassen)
# --------------------------------
@st.cache_resource
def load_labels():
    import json
    import urllib.request

    url = "https://storage.googleapis.com/public-datasets-lila/iNat21/iNat21_labels.json"
    try:
        with urllib.request.urlopen(url) as f:
            labels = json.load(f)
        return labels
    except:
        return None

labels = load_labels()

# --------------------------------
# Image preprocessing
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
    st.image(image, caption="Dein Bild", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top5 = torch.topk(probs, 5)

    st.subheader("🔍 Top Ergebnisse")

    for score, idx in zip(top5.values, top5.indices):

        label = labels[idx] if labels else f"Class {idx}"
        confidence = round(float(score) * 100, 2)

        st.write(f"**{label}** — {confidence}%")

        # einfache Giftigkeitsprüfung
        for key in TOXICITY_DB:
            if key.lower() in label.lower():
                st.warning(TOXICITY_DB[key])

# --------------------------------
# Hinweis
# --------------------------------
st.info(
    "Modell basiert auf iNaturalist Biodiversitätsdaten. "
    "Ergebnisse sind näher an echten Pflanzenarten als ImageNet-Modelle."
)
