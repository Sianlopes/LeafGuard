import hashlib
import json
import streamlit as st
import plotly.express as px
from src.ui import setup, choose_model, model_help
from src.preprocessing import load_image
from src.inference import Predictor, ModelUnavailableError
from src.common import ROOT, read_json

setup("Disease detection")
st.markdown('<div class="eyebrow">Leaf image analysis</div>', unsafe_allow_html=True)
st.title("What is your leaf showing?")
st.write("Upload one leaf, then analyse it to see the model's assessment and alternatives.")
model_dir = choose_model()
ready = all((model_dir / name).exists() for name in ("best.keras", "class_names.json", "inference_config.json"))
if not ready:
    model_help()
with st.expander("Tips for a useful image"):
    st.write("Use a sharp, well-lit photograph of one leaf. Fill most of the frame with the leaf, keep the background simple, and avoid filters. Supported uploads: JPG, JPEG and PNG up to 10 MB.")
left, right = st.columns([1, 1.25], gap="large")
with left:
    uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    example_manifest = ROOT / "data/examples/manifest.json"
    examples = read_json(example_manifest)["images"] if example_manifest.exists() else []
    example_name = st.selectbox("Or try an example leaf", ["Choose an image"] + [row["name"] for row in examples], disabled=uploaded is not None) if examples else "Choose an image"
    source_bytes = uploaded.getvalue() if uploaded is not None else None
    if source_bytes is None and example_name != "Choose an image":
        chosen_example = next(row for row in examples if row["name"] == example_name)
        source_bytes = (ROOT / "data/examples" / chosen_example["file"]).read_bytes()
        st.caption("Example: PlantVillage · Mohanty, Hughes & Salathé (2016) · CC BY-SA 3.0. Controlled images may differ from field photographs.")
    image = None
    if source_bytes is not None:
        try:
            image = load_image(source_bytes)
            st.image(image, caption="Selected leaf image", width="stretch")
        except ValueError as exc:
            st.error(str(exc))
    heatmap_enabled = st.checkbox("Show Grad-CAM explanation", value=False)
    threshold = st.slider("Minimum confidence", min_value=.5, max_value=.99, value=.8, step=.01, help="Can only make a validated rejection policy stricter. Confidence alone cannot validate an unknown image.")
    analyse = st.button("Analyse Leaf", type="primary", disabled=image is None or not ready, width="stretch")
    st.caption("Images are processed in this running app. They are not added to the training dataset.")
signature = hashlib.sha256(source_bytes + str(model_dir).encode() + str(threshold).encode()).hexdigest() if source_bytes is not None and image is not None else None
if analyse:
    try:
        with st.spinner("Examining leaf features…"):
            # Streamlit resource cache is keyed by artifact modification time.
            @st.cache_resource(show_spinner=False)
            def cached_predictor(directory, model_time, calibration_time):
                return Predictor(directory)
            predictor = cached_predictor(str(model_dir), (model_dir / "best.keras").stat().st_mtime_ns, (model_dir / "calibration.json").stat().st_mtime_ns if (model_dir / "calibration.json").exists() else 0)
            result = predictor.predict(image, threshold)
            st.session_state["leaf_analysis"] = {"signature": signature, "result": result}
    except (ModelUnavailableError, ValueError, RuntimeError, OSError) as exc:
        st.session_state.pop("leaf_analysis", None)
        with right:
            st.error(str(exc))
saved = st.session_state.get("leaf_analysis")
with right:
    if not saved or saved["signature"] != signature or signature is None:
        if not analyse:
            with st.container(border=True):
                st.markdown("### Your assessment will appear here")
                st.write("Upload an image and select **Analyse Leaf**. You'll see the leading categories, model confidence, and any available guidance.")
    else:
        result = saved["result"]
        if result["uncertain"]:
            st.warning(result["message"])
            if not result["rejection_validated"]:
                st.caption("Rejection calibration has not passed an independent audit. The model rankings below are unverified candidates.")
        else:
            st.success("Analysis complete · AI-based suggestion")
            st.subheader(result["plant"])
            st.markdown(f"### {result['disease']}")
        a, b = st.columns(2)
        a.metric("Leading confidence", f"{result['confidence']:.1%}")
        b.metric("Model", "EfficientNetB0" if result["model"] == "efficientnet" else "Custom CNN")
        st.caption("Temperature-scaled model probability" if result["calibrated"] else "Uncalibrated softmax probability; high values can occur on unrelated images.")
        st.markdown("#### Top model candidates")
        top = result["top_three"]
        figure = px.bar(x=[row["confidence"] for row in top], y=[f"{row['plant']} · {row['disease']}" for row in top], orientation="h", text=[f"{row['confidence']:.1%}" for row in top], color_discrete_sequence=["#43835b"])
        figure.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%", title="Model probability"), yaxis=dict(title="", autorange="reversed"), margin=dict(l=0, r=30, t=10, b=10), height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(figure, width="stretch")
        st.download_button("Download analysis JSON", json.dumps(result, indent=2), file_name="leafguard_analysis.json", mime="application/json")
        info = result["disease_info"]
        if info:
            st.divider()
            st.markdown("#### Understanding this category")
            st.write(info["description"])
            with st.expander("Common symptoms", expanded=True):
                for symptom in info["symptoms"]:
                    st.write(f"• {symptom}")
            with st.expander("General prevention guidance", expanded=True):
                for guidance in info["prevention"]:
                    st.write(f"• {guidance}")
            for source in info["sources"]:
                st.markdown(f"[Agricultural reference]({source})")
        if heatmap_enabled:
            try:
                from src.gradcam import gradcam
                predictor = Predictor(model_dir)
                with st.spinner("Calculating class influence…"):
                    overlay, _ = gradcam(predictor, image, result["predicted_index"])
                st.image(overlay, caption="Grad-CAM · influence on the leading model candidate", width="stretch")
                st.caption("Highlights regions that influenced this class score. It does not prove that disease is located in those regions.")
            except (ValueError, RuntimeError, OSError) as exc:
                st.info(f"Grad-CAM unavailable: {exc}")
st.divider()
st.caption("AI-based suggestions, not confirmed laboratory diagnoses. Unknown plants, mixed symptoms and field backgrounds can mislead the model. Consult local agricultural expertise before treatment decisions.")
