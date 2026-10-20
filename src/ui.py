"""Shared Streamlit presentation and metadata access."""
from html import escape
from pathlib import Path
import streamlit as st
from src.common import ROOT, read_json, model_directory


def icon(name, size=20):
    """Small, consistent outline icons; decorative alongside visible labels."""
    paths = {
        "leaf": '<path d="M20 4c-8-1-15 2-15 9a6 6 0 0 0 6 6c7 0 10-7 9-15Z"/><path d="M4 21 15 10M9 16v-5m0 5h5"/>',
        "scan": '<path d="M8 3H4a1 1 0 0 0-1 1v4m13-5h4a1 1 0 0 1 1 1v4M3 16v4a1 1 0 0 0 1 1h4m8 0h4a1 1 0 0 0 1-1v-4M7 12h10"/><path d="M12 7v10"/>',
        "image": '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/>',
        "review": '<path d="M9 5H5a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-4"/><rect x="9" y="2" width="6" height="6" rx="2"/><path d="m8 15 3 3 5-6"/>',
    }
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>'


def page_heading(eyebrow, title, description):
    st.markdown(f'<div class="page-heading"><div class="eyebrow">{escape(eyebrow)}</div><h1>{escape(title)}</h1><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def panel_heading(number, title, description=None):
    st.markdown(f'<div class="panel-heading"><span class="panel-number">{escape(number)}</span><h3>{escape(title)}</h3></div>' + (f'<p class="panel-description">{escape(description)}</p>' if description else ''), unsafe_allow_html=True)


def chart_style(figure):
    """Keep Plotly panels in the same visual language as the rest of the app."""
    figure.update_layout(font=dict(family="Inter, Segoe UI, sans-serif", color="#617068", size=12), title_font=dict(size=16, color="#202e28"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12, x=0), hoverlabel=dict(bgcolor="#ffffff", font_color="#202e28"))
    figure.update_xaxes(gridcolor="#e7ece4", zerolinecolor="#e7ece4")
    figure.update_yaxes(gridcolor="#e7ece4", zerolinecolor="#e7ece4")
    return figure


def setup(title):
    st.set_page_config(page_title=f"{title} · LeafGuard AI", page_icon=":material/eco:", layout="wide", initial_sidebar_state="auto")
    st.markdown(f'<style>{(ROOT / "assets/styles.css").read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f'<div class="brand"><div class="brand-mark">{icon("leaf", 23)}</div><div><div class="brand-name">LeafGuard <span>AI</span></div><div class="brand-subtitle">A closer look at plant health</div></div></div><div class="nav-label">Workspace</div>', unsafe_allow_html=True)
        with st.container(key="sidebar_navigation"):
            for page, label, symbol in [
                ("app.py", "Overview", "space_dashboard"),
                ("pages/1_Detect_Disease.py", "Disease detection", "frame_inspect"),
                ("pages/2_Supported_Plants.py", "Supported plants", "eco"),
                ("pages/3_Model_Performance.py", "Model performance", "monitoring"),
                ("pages/4_About.py", "About the project", "auto_stories"),
            ]:
                if label == title:
                    with st.container(key="active_navigation"):
                        st.page_link(page, label=label, icon=f":material/{symbol}:", width="stretch")
                else:
                    st.page_link(page, label=label, icon=f":material/{symbol}:", width="stretch")
        st.markdown('<div class="sidebar-note"><strong>Built for informed observation</strong><br>Academic research · PlantVillage<br>CNN + EfficientNetB0<br><br>AI suggestions support observation. They do not confirm a laboratory diagnosis.</div>', unsafe_allow_html=True)


def choose_model():
    directories = sorted({p.parent for p in (ROOT / "models").glob("*/inference_config.json")} | {model_directory()})
    current = model_directory().resolve()
    values = sorted({str(p.resolve()) for p in directories})
    return Path(st.selectbox("Analysis model", values, index=values.index(str(current)) if str(current) in values else 0, format_func=lambda p: {"efficientnet": "EfficientNetB0", "cnn": "Custom CNN"}.get(Path(p).name, Path(p).name), width=340))


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
