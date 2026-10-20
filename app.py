"""LeafGuard AI home. Run with: streamlit run app.py."""
import streamlit as st
from html import escape
from src.ui import setup, installed_classes, icon
from src.common import parse_label, model_directory

setup("Overview")
st.markdown("""<div class="hero"><div><div class="eyebrow">Observe · Understand · Protect</div>
<h1>Healthier plants.<br><span>A closer look.</span></h1>
<p>Multi-Plant Disease Detection Using Deep Learning</p>
<p>Explore a leaf image with computer vision. Understand visible symptoms, compare possible categories, and take a more informed next step.</p></div>
<div class="hero-art" role="img" aria-label="Botanical illustration of a leaf with an observation frame">
<svg viewBox="0 0 330 300" fill="none" aria-hidden="true">
<circle cx="173" cy="145" r="121" stroke="#d6e1cc"/><circle cx="173" cy="145" r="94" stroke="#d6e1cc" stroke-dasharray="2 7"/>
<path d="M167 238C89 207 97 100 241 42c28 121-10 191-74 196Z" fill="#718e5e"/>
<path d="M167 238c-13-54 14-131 74-196-8 102-28 174-74 196Z" fill="#527748"/>
<path d="M145 271C150 207 187 127 235 53" stroke="#33583e" stroke-width="3" stroke-linecap="round"/>
<path d="m159 224-34-53m45 24-43-60m57 31-34-55m53 21-23-42m-20 125 61-31m-47 3 63-43m-50 15 49-37" stroke="#bbcc9f" stroke-width="1.2" stroke-linecap="round"/>
<path d="M76 103V83h20m171 0h20v20M76 205v20h20m171 0h20v-20" stroke="#648060" stroke-width="1.5"/>
<path d="M65 154h228" stroke="#486d45" stroke-opacity=".22" stroke-dasharray="4 4"/>
<circle cx="198" cy="151" r="5" fill="#edf2e7" stroke="#486d45"/>
</svg><div class="hero-art-label"><strong>Plant vision, with perspective.</strong><span>Observe patterns. Understand uncertainty.</span></div></div></div>""", unsafe_allow_html=True)
with st.container(key="home_actions", horizontal=True, gap="small"):
    with st.container(key="primary_action"):
        st.page_link("pages/1_Detect_Disease.py", label="Start detection", icon=":material/arrow_forward:", icon_position="right", width="stretch")
    with st.container():
        st.page_link("pages/2_Supported_Plants.py", label="Explore supported plants", icon=":material/eco:", width="stretch")
classes, source = installed_classes()
crops = {parse_label(c)[0] for c in classes}
a, b, c = st.columns(3)
a.metric("Plant species", len(crops) if crops else "—")
b.metric("Classification categories", len(classes) if classes else "—")
c.metric("Model status", "Installed" if (model_directory() / "best.keras").exists() else "Training needed")
st.caption(f"Catalogue source: {source} · Counts reflect actual prepared data or installed model metadata.")
st.markdown('<div class="section-intro"><span>THE WORKFLOW</span><h2>A little clarity, in three steps.</h2></div>', unsafe_allow_html=True)
columns = st.columns(3, gap="medium")
for column, number, name, title, detail in zip(columns, ["01 / CAPTURE", "02 / ANALYSE", "03 / REVIEW"], ["image", "scan", "review"], ["Bring a clear image", "Compare visual patterns", "Choose the next step"], ["Photograph one leaf in even light. Keep the surface in focus and avoid heavy shadows.", "A trained neural network compares features with its supported PlantVillage categories.", "Review alternatives, uncertainty and curated guidance. Confirm concerning symptoms with an agricultural specialist."]):
    with column:
        st.markdown(f'<div class="step-card"><div class="step-top"><div class="icon-tile">{icon(name)}</div><span class="step-number">{number}</span></div><h3>{escape(title)}</h3><p>{escape(detail)}</p></div>', unsafe_allow_html=True)
st.divider()
left, right = st.columns([1.3, 1], gap="large")
with left:
    st.subheader("Small changes deserve attention.")
    st.write("Regular observation helps growers notice spreading symptoms earlier and seek timely advice. Similar spots, discoloration and curling can have several causes; an image alone cannot establish which pathogen is present.")
with right:
    with st.container(border=True, key="scope_card"):
        st.markdown("**Designed for a defined set of crops**")
        st.write("This project uses PlantVillage categories. It does not identify every plant or disease, and laboratory-style images differ from photographs taken in the field.")
        st.caption("Predictions and performance panels use the installed model and saved evaluation artifacts.")
