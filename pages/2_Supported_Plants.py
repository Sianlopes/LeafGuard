import streamlit as st
import pandas as pd
from src.ui import setup, dataset_metadata, installed_classes
from src.common import parse_label

setup("Supported plants")
st.markdown('<div class="eyebrow">The training catalogue</div>', unsafe_allow_html=True)
st.title("Know the model's growing ground.")
st.write("Browse the plant-health categories available in the actual project metadata.")
metadata = dataset_metadata()
classes, source = installed_classes()
if metadata:
    table = metadata[1].copy()
    if not metadata[0].get("complete_physical_leaf_verification", True):
        st.warning(f"Publisher splits with duplicate auditing: {metadata[0]['unverified_physical_leaf_records']:,} records have no verified physical-leaf ID. Complete physical-leaf separation cannot be verified.")
    st.caption(f"Prepared dataset · {metadata[0]['images']:,} usable images · {metadata[0]['excluded']:,} excluded during audit")
elif classes:
    table = pd.DataFrame({"label": classes, "train": [None] * len(classes)})
    st.info("Showing installed model classes. Prepare dataset metadata to display actual training image counts.")
else:
    st.info("The plant catalogue is not available yet. Download and prepare the dataset to discover its species, categories and image counts.")
    st.code("python -m src.dataset download\npython -m src.dataset prepare", language="bash")
    st.stop()
table["Plant"] = table["label"].map(lambda c: parse_label(c)[0])
table["Category"] = table["label"].map(lambda c: parse_label(c)[1])
query = st.text_input("Search plants or categories", placeholder="Try potato, apple, rust…")
filtered = table[table["Plant"].str.contains(query, case=False, regex=False) | table["Category"].str.contains(query, case=False, regex=False)]
st.caption(f"{filtered['Plant'].nunique()} plants · {len(filtered)} categories match")
for plant in sorted(filtered["Plant"].unique()):
    rows = table[table["Plant"] == plant]
    with st.expander(f"{plant} · {len(rows)} classifications", expanded=bool(query)):
        a, b, c = st.columns(3)
        a.metric("Training images", f"{int(rows['train'].sum()):,}" if metadata else "Unavailable")
        b.metric("Disease / pest categories", int((rows["Category"].str.lower() != "healthy").sum()))
        c.metric("Healthy category", "Available" if (rows["Category"].str.lower() == "healthy").any() else "Not provided")
        show = rows.rename(columns={"train": "Training images", "validation": "Validation", "calibration": "Calibration", "test": "Test"})
        columns = ["Category", "Training images"] + (["Validation", "Calibration", "Test"] if metadata else [])
        st.dataframe(show[columns], hide_index=True, width="stretch")
st.caption("Support for a category means it appears in the metadata. It does not establish field accuracy or coverage of all diseases affecting that crop.")
