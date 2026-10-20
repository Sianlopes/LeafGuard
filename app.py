"""LeafGuard AI home. Run with: streamlit run app.py."""
import streamlit as st
from src.ui import setup, installed_classes, dataset_metadata
from src.common import parse_label, model_directory

setup("Overview")
st.markdown("""<div class="hero"><div class="eyebrow">Observe · Understand · Protect</div>
<h1>Healthier plants start<br>with a closer look.</h1>
<p><strong>LeafGuard AI</strong><br>Multi-Plant Disease Detection Using Deep Learning</p>
<p>Explore a leaf image with computer vision. Compare possible plant-health categories,
understand visible symptoms, and learn when an expert's second look matters.</p></div>""", unsafe_allow_html=True)
left, right = st.columns([2, 1])
with left:
    st.page_link("pages/1_Detect_Disease.py", label="Start Detection →", icon="🔎", width="stretch")
with right:
    st.page_link("pages/2_Supported_Plants.py", label="Explore supported plants", icon="🌱", width="stretch")
st.write("")
classes, source = installed_classes()
crops = {parse_label(c)[0] for c in classes}
a, b, c = st.columns(3)
a.metric("Plant species", len(crops) if crops else "—")
b.metric("Classification categories", len(classes) if classes else "—")
c.metric("Model status", "Installed" if (model_directory() / "best.keras").exists() else "Training needed")
st.caption(f"Catalogue source: {source}. Counts become available from actual prepared data or installed model metadata.")
st.write("")
st.subheader("From leaf image to informed observation")
columns = st.columns(3, gap="large")
for column, number, title, detail in zip(columns, ["01 / CAPTURE", "02 / ANALYSE", "03 / REVIEW"], ["Bring a clear image", "Compare visual patterns", "Choose the next step"], ["Photograph one leaf in even light. Keep the surface in focus and avoid heavy shadows.", "A trained neural network compares features with its supported PlantVillage categories.", "Review alternatives, uncertainty and curated guidance. Confirm concerning symptoms with an agricultural specialist."]):
    with column:
        st.markdown(f'<div class="step">{number}</div>', unsafe_allow_html=True)
        st.markdown(f"#### {title}")
        st.write(detail)
st.divider()
left, right = st.columns([1.3, 1], gap="large")
with left:
    st.subheader("Small changes deserve attention.")
    st.write("Regular observation helps growers notice spreading symptoms earlier and seek timely advice. Similar spots, discoloration and curling can have several causes; an image alone cannot establish which pathogen is present.")
with right:
    with st.container(border=True):
        st.markdown("**Designed for a defined set of crops**")
        st.write("This project uses PlantVillage categories. It does not identify every plant or disease, and laboratory-style images differ from photographs taken in the field.")
        st.caption("No predictions or performance metrics are generated until actual trained artifacts are available.")
