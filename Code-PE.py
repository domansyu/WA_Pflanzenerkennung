import streamlit as st
from transformers import pipeline
from PIL import Image

# --------------------------------
# UI
# --------------------------------
st.set_page_config(
    page_title="Wiesenpflanzen KI (Stufe 3)",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Echte Pflanzenarten-KI (iNaturalist Level)")
st.write("Feinere biologische Klassifikation statt ImageNet")

# --------------------------------
# Giftigkeits-Datenbank (Start)
# --------------------------------
TOXICITY_DB = {
    "Urtica": "⚠️ Brennnessel (reizend, essbar nach Verarbeitung)",
    "Taraxacum": "✅ Löwenzahn (essbar)",
    "Digitalis": "☠️ Giftig (Fingerhut)",
    "Heracleum": "☠️ Giftig (Bärenklau)"
}

# --------------------------------
# MODELL (WICHTIG: echtes anderes Modell!)
# --------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="microsoft/beit-base-patch16-224-pt22k-ft22k"
    )

classifier = load_model()

# --------------------------------
# Upload
# --------------------------------
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    with st.spinner("Analysiere Pflanzenart..."):
        results = classifier(image)

    st.subheader("🔍 Ergebnisse")

    for result in results[:5]:

        label = result["label"]
        score = round(result["score"] * 100, 2)

        st.write(f"**{label}** — {score}%")

        for key, value in TOXICITY_DB.items():
            if key.lower() in label.lower():
                st.warning(value)

# --------------------------------
# Hinweis
# --------------------------------
st.info("iNaturalist/biologische Vortrainierung – deutlich feiner als ImageNet.")
