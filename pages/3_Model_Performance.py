from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.ui import setup, page_heading, chart_style
from src.common import ROOT, read_json

setup("Model performance")
page_heading("Measured, not assumed", "Evidence behind the assessment.", "A transparent view of saved held-out test results. Model selection uses validation loss; these panels show the resulting evidence.")
metrics_dir = ROOT / "reports/metrics"
available = [name for name in ("efficientnet", "cnn") if (metrics_dir / f"{name}.json").exists()]
if not available:
    st.info("Evaluation results are unavailable. Train a model and run evaluate.py to populate this page. No experimental results have been invented.")
    st.code("python evaluate.py --model-dir models/cnn\npython evaluate.py --model-dir models/efficientnet", language="bash")
    st.stop()
name = st.selectbox("Evaluated model", available, format_func=lambda value: "EfficientNetB0" if value == "efficientnet" else "Custom CNN", width=340)
try:
    metrics = read_json(metrics_dir / f"{name}.json")
    if not metrics.get("complete_physical_leaf_verification", True):
        st.warning("The publisher's split was audited for duplicates, but some physical-leaf IDs are unavailable. These scores do not establish complete physical-leaf independence or field accuracy.")
    columns = st.columns(4)
    for column, key, title in zip(columns, ["test_accuracy", "precision", "recall", "macro_f1"], ["Test accuracy", "Macro precision", "Macro recall", "Macro F1"]):
        column.metric(title, f"{metrics[key]:.2%}")
    st.caption(f"{metrics['test_images']:,} held-out images · weighted F1 {metrics['weighted_f1']:.4f} · checkpoint {metrics['model_sha256'][:12]}")
    accuracy_tab, confusion_tab, class_tab, comparison_tab = st.tabs(["Training curves", "Confusion matrix", "Per-class results", "Model comparison"])
    with accuracy_tab:
        # History copied into evaluation output to support custom model locations.
        history_path = metrics_dir / f"{name}_history.json"
        if history_path.exists():
            history = read_json(history_path)
            left, right = st.columns(2)
            for column, suffix, title in ((left, "accuracy", "Accuracy"), (right, "loss", "Loss")):
                chart = go.Figure()
                for key, label, color in ((suffix, "Training", "#24694e"), (f"val_{suffix}", "Validation", "#a48047")):
                    chart.add_trace(go.Scatter(x=list(range(1, len(history[key]) + 1)), y=history[key], name=label, line=dict(color=color, width=2.5)))
                chart.update_layout(title=title, xaxis_title="Epoch", yaxis_title=title, height=350, margin=dict(l=0, r=0, t=45, b=0))
                column.plotly_chart(chart_style(chart), width="stretch")
        else:
            st.info("Saved training history is unavailable; rerun evaluation to include it.")
    with confusion_tab:
        matrix = pd.read_csv(metrics_dir / f"{name}_confusion_matrix.csv", index_col=0)
        chart = px.imshow(matrix, color_continuous_scale="Greens", labels=dict(x="Predicted class", y="Actual class", color="Images"), aspect="auto")
        chart.update_layout(height=max(650, len(matrix) * 22))
        st.plotly_chart(chart_style(chart), width="stretch")
    with class_tab:
        per_class = pd.read_csv(metrics_dir / f"{name}_per_class.csv")
        st.plotly_chart(chart_style(px.bar(per_class, x="recall", y="class", orientation="h", range_x=[0, 1], color_discrete_sequence=["#4b7b59"], height=max(400, len(per_class) * 26))), width="stretch")
        st.dataframe(per_class, hide_index=True, width="stretch")
        st.download_button("Download classification report CSV", per_class.to_csv(index=False), file_name=f"{name}_per_class.csv", icon=":material/download:")
    with comparison_tab:
        path = metrics_dir / "comparison.csv"
        if path.exists():
            comparison = pd.read_csv(path)
            st.plotly_chart(chart_style(px.bar(comparison.melt(id_vars="model", var_name="Metric", value_name="Score"), x="model", y="Score", color="Metric", barmode="group", range_y=[0, 1], color_discrete_sequence=["#24694e", "#92ad86", "#a48047"])), width="stretch")
            st.dataframe(comparison, hide_index=True, width="stretch")
        else:
            st.info("Evaluate both architectures on the same manifest to create a valid comparison.")
    st.caption("Test data must remain untouched during model and threshold selection. Repeated test-driven tuning invalidates this held-out interpretation. PlantVillage test scores do not establish field reliability.")
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.warning(f"Evaluation artifacts are incomplete or invalid: {exc}. Rerun evaluate.py.")
