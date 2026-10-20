import streamlit as st
from src.ui import setup

setup("About the project")
st.markdown('<div class="eyebrow">Research & methodology</div>', unsafe_allow_html=True)
st.title("A practical study in plant vision.")
st.write("LeafGuard AI is an academic project exploring multi-crop leaf classification with deep learning, reproducible evaluation and explicit uncertainty.")
left, right = st.columns([1.5, 1], gap="large")
with left:
    st.subheader("The problem")
    st.write("Visible leaf symptoms can guide further investigation, but diseases, pests and environmental stress often look similar. This project studies whether image models can distinguish the defined plant-health categories in PlantVillage.")
    st.subheader("The objectives")
    st.write("Train a custom CNN and an ImageNet-pretrained EfficientNetB0 independently; compare held-out performance; expose real predictions through an accessible interface; and clearly reject uncertain assessments.")
    st.subheader("Two approaches to visual learning")
    with st.expander("Custom CNN baseline", expanded=True):
        st.write("Three convolution–batch-normalisation–ReLU–pooling blocks learn local patterns, followed by a final convolution, global average pooling, dropout and a softmax classifier. Internal rescaling converts RGB pixels to [0,1].")
    with st.expander("EfficientNetB0 transfer learning", expanded=True):
        st.write("ImageNet features are initially frozen while the new classification head trains. Optional fine-tuning unfreezes selected deeper layers at a lower learning rate while keeping batch normalisation statistics frozen. Keras EfficientNet includes its own preprocessing and receives RGB values in [0,255].")
    st.subheader("From data to evidence")
    st.write("The preparation pipeline discovers classes, validates images and groups available physical-leaf IDs, exact pixel duplicates and conservative near-duplicates. When source groups cross splits, held-out observations are retained and matching training observations are excluded. Some source records lack physical-leaf IDs, so complete physical-leaf independence cannot be verified. Augmentation applies only to training. Validation loss selects checkpoints; reserved calibration groups support temperature scaling and rejection checks; the test partition supports final evaluation.")
    st.write("Evaluation reports accuracy, macro precision and recall, macro and weighted F1, per-class results and a confusion matrix. Optional Grad-CAM highlights regions influencing a class score, without establishing where disease is located.")
with right:
    with st.container(border=True):
        st.markdown("### Dataset")
        st.write("PlantVillage contains approximately 54,000 original leaf images across 14 crop species and 38 categories in its commonly used color release. The app's catalogue and counts come from actual prepared metadata.")
        st.markdown("[PlantVillage source](https://huggingface.co/datasets/mohanty/PlantVillage)")
        st.caption("Credit: Mohanty, Hughes & Salathé (2016). Dataset license: CC BY-SA 3.0; retain source attribution when sharing data.")
    st.subheader("Practical limitations")
    st.write("• Controlled backgrounds differ from field images.\n\n• Confidence can be high on unknown objects.\n\n• Physical-leaf separation cannot be verified for every source record.\n\n• Public OOD sources lack capture-session IDs; the audit measures rejection on the supplied disjoint images and species.\n\n• A healthy category does not rule out latent infection.\n\n• Unseen crops, mixed disease and severity are outside the trained task.")
    st.subheader("Future improvements")
    st.write("Collect independent field data, audit performance across locations and cameras, strengthen unknown-image detection, and involve agronomists in external validation.")
st.divider()
st.markdown("[Original PlantVillage study](https://doi.org/10.3389/fpls.2016.01419) · [EfficientNet paper](https://arxiv.org/abs/1905.11946) · [Grad-CAM paper](https://arxiv.org/abs/1610.02391)")
