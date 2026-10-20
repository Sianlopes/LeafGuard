"""Shared Streamlit presentation and metadata access."""
import os
from pathlib import Path
import streamlit as st
from src.common import ROOT, read_json, model_directory, parse_label


def setup(title):
    st.set_page_config(page_title=f"{title} · LeafGuard AI", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>
    .block-container{max-width:1240px;padding-top:2.5rem;padding-bottom:3rem}
    h1,h2,h3{letter-spacing:-.035em} h1{font-size:2.7rem!important;line-height:1.12!important}
    [data-testid="stSidebar"]{border-right:1px solid #dfe9de}
    [data-testid="stMetric"]{background:#eef5ed;border:1px solid #dfe9de;border-radius:15px;padding:18px}
    [data-testid="stMetricLabel"]{color:#4b6657}
    .eyebrow{font-size:.75rem;text-transform:uppercase;letter-spacing:.16em;color:#477357;font-weight:700;margin-bottom:16px}
    .hero{padding:38px 40px;background:linear-gradient(115deg,#e8f1e3,#f5f8ef);border:1px solid #d5e2ce;border-radius:24px;margin-bottom:24px}
    .hero p{max-width:740px;font-size:1.08rem;line-height:1.7;color:#4b6657}
    .note{font-size:.9rem;color:#57715f;line-height:1.6}
    .step{font-family:monospace;color:#6e916d;font-size:1.1rem;margin-bottom:12px}
    [data-testid="stFileUploader"]{border-radius:16px}
    .stButton button,.stDownloadButton button{border-radius:10px;min-height:2.7rem}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## 🌿 LeafGuard AI")
        st.caption("A closer look at plant health")
        st.divider()
        st.page_link("app.py", label="Overview", icon="🏡")
        st.page_link("pages/1_Detect_Disease.py", label="Disease detection", icon="🔎")
        st.page_link("pages/2_Supported_Plants.py", label="Supported plants", icon="🌱")
        st.page_link("pages/3_Model_Performance.py", label="Model performance", icon="📊")
        st.page_link("pages/4_About.py", label="About the project", icon="📖")
        st.divider()
        st.caption("Academic research project\n\nPlantVillage · CNN + EfficientNetB0")
        st.caption("AI suggestions support observation. They do not confirm a laboratory diagnosis.")


def choose_model():
    directories = sorted({p.parent for p in (ROOT / "models").glob("*/inference_config.json")} | {model_directory()})
    current = model_directory().resolve()
    values = sorted({str(p.resolve()) for p in directories})
    return Path(st.selectbox("Model artifacts", values, index=values.index(str(current)) if str(current) in values else 0, format_func=lambda p: Path(p).name))


def dataset_metadata():
    processed = ROOT / "data/processed"
    if not (processed / "metadata.json").exists():
        return None
    try:
        import pandas as pd
        metadata = read_json(processed / "metadata.json")
        summary = pd.read_csv(processed / "summary.csv")
        return metadata, summary
    except (ValueError, OSError, KeyError):
        st.warning("Dataset metadata is incomplete. Run the dataset preparation command again.")
        return None


def installed_classes():
    directory = model_directory()
    file = directory / "class_names.json"
    if file.exists():
        try:
            classes = read_json(file)
            if isinstance(classes, list) and all(isinstance(c, str) and "___" in c for c in classes):
                return classes, "Installed model"
        except (ValueError, OSError):
            pass
    metadata = dataset_metadata()
    return (metadata[0]["classes"], "Prepared dataset") if metadata else ([], "Dataset not prepared")


def model_help(message=None):
    st.info(message or "Install a trained model to analyse leaves. The app is ready for artifacts from local training or Google Colab.")
    with st.expander("Prepare the model", expanded=True):
        st.code("python -m src.dataset download\npython -m src.dataset prepare\npython train.py --model efficientnet\npython evaluate.py --model-dir models/efficientnet", language="bash")
        st.write("Copy the complete Colab `models/efficientnet/` folder into this project's `models/efficientnet/`, then restart Streamlit. Run calibration with independent OOD tuning and audit images to enable accepted suggestions.")
        st.caption("Full instructions: README.md and notebooks/training_colab.ipynb")
