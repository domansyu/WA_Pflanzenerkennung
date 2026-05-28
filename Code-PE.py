import streamlit as st
from transformers import pipeline
from PIL import Image

# --------------------------------
# UI
# --------------------------------
st.set_page_config(
    page_title="Pflanzen KI",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Pflanzen KI (stabile Version)")
st.write("Lokale Bildklassifikation ohne API")

# --------------------------------
# Modell (STABIL)
# --------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="facebook/deit-base-patch16-224"
    )

classifier = load_model()

# --------------------------------
# Upload
# --------------------------------
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    results = classifier(image)

    st.subheader("🔍 Ergebnisse")

    for r in results[:5]:
        st.write(f"**{r['label']}** — {round(r['score']*100, 2)}%")

# --------------------------------
# Hinweis
# --------------------------------
st.info("Stabile Version: funktioniert sicher, aber nur grobe Klassifikation.")
# --------------------------------
st.info("iNaturalist/biologische Vortrainierung – deutlich feiner als ImageNet.")
