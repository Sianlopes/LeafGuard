"""Generate the checked-in, output-free Colab notebook using standard JSON."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def markdown(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
    def code(text):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)})

    markdown("""# LeafGuard AI · Complete PlantVillage experiment

Use **Runtime → Change runtime type → GPU** for practical full-dataset training.
This notebook uses the project modules, not a second divergent implementation.
It trains both models on all discovered color categories. No sample accuracy
values are invented. The repository includes a completed CPU experiment; the
cells below create a separate experiment. Expect a large download and a long
training run. Use a Python 3.11–3.13 runtime for the pinned TensorFlow version.

First run `python scripts/package_source.py` locally, then upload that source
archive below. Alternatively provide your own GitHub repository URL.
""")
    code('''from pathlib import Path
import os, subprocess, sys, zipfile

PROJECT = Path("/content/LeafGuard-AI")
REPOSITORY_URL = "https://github.com/Sianlopes/LeafGuard.git"  # Set blank to upload leafguard_source.zip instead.
if not PROJECT.exists():
    if REPOSITORY_URL:
        subprocess.run(["git", "clone", REPOSITORY_URL, str(PROJECT)], check=True)
    else:
        from google.colab import files
        uploaded = files.upload()  # Select leafguard_source.zip
        names = [name for name in uploaded if name.endswith(".zip")]
        if len(names) != 1:
            raise ValueError("Upload exactly one project source ZIP.")
        PROJECT.mkdir(parents=True)
        with zipfile.ZipFile(names[0]) as archive:
            for member in archive.infolist():
                target = (PROJECT / member.filename).resolve()
                if not target.is_relative_to(PROJECT.resolve()):
                    raise ValueError("Unsafe ZIP member.")
            archive.extractall(PROJECT)
os.chdir(PROJECT)
assert Path("train.py").exists(), "Source archive must contain project files at its root."
print("Project:", PROJECT, "Python:", sys.version)
''')
    code('''# Install before importing TensorFlow/NumPy. If Colab requests a runtime restart,
# restart once and rerun the source-location cell before continuing below.
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
''')
    code('''import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())
def run(*arguments):
    subprocess.run([sys.executable, *arguments], check=True)
''')
    markdown("""## Download, audit and inspect the complete color dataset

The source revision, official test split and physical leaf map are preserved.
Validation and calibration groups come only from the source training pool.
If networking fails, follow README manual import instructions; no substitute
dataset is created. You can set DATA_ROOT to a mounted Drive folder containing
images plus verified groups.csv and set DOWNLOAD=False.

**Known source limitation:** 11,790 of 54,305 source records lack verified
physical-leaf identities after authentic legacy aliases. The approved publisher
policy preserves source assignments and audits available leaf IDs, exact hashes
and conservative near-duplicates, without inventing physical-leaf IDs or dropping
categories. Complete physical-leaf independence cannot be verified. Strict policy
requires repaired metadata. This limitation must accompany reported scores.
""")
    code('''DATA_ROOT = "data/raw"
DOWNLOAD = True
if DOWNLOAD:
    run("-m", "src.dataset", "download", "--data-dir", DATA_ROOT, "--group-policy", "publisher")
run("-m", "src.dataset", "prepare", "--data-dir", DATA_ROOT, "--output", "data/processed", "--group-policy", "publisher")
summary = pd.read_csv("data/processed/summary.csv")
display(summary)
summary.set_index("label")[["train", "validation", "calibration", "test"]].plot.barh(stacked=True, figsize=(12, 13))
plt.title("Actual usable image counts by category and split")
plt.tight_layout()
plt.show()
''')
    markdown("""## Train both architectures independently

Preprocessing, augmentation, deterministic seeds and callbacks live in the same
modules used locally. RGB values remain in [0,255]; architecture-specific
rescaling is embedded in the networks. Augmentation applies only to training.
The checkpoint directory must be new; choose new output folders for a new run
rather than overwriting an experiment.
""")
    code('''EPOCHS = 15
FINE_TUNE_EPOCHS = 10
BATCH_SIZE = 32  # Reduce to 16 or 8 if GPU memory is insufficient.
MODEL_DIRS = {name: f"models/{name}_colab" for name in ("cnn", "efficientnet")}
run("train.py", "--model", "cnn", "--output", MODEL_DIRS["cnn"], "--epochs", str(EPOCHS), "--batch-size", str(BATCH_SIZE))
''')
    code('''run("train.py", "--model", "efficientnet", "--output", MODEL_DIRS["efficientnet"], "--epochs", str(EPOCHS),
    "--fine-tune-epochs", str(FINE_TUNE_EPOCHS), "--batch-size", str(BATCH_SIZE))
''')
    markdown("""## Final held-out evaluation

Choose final hyperparameters from validation evidence before this step. These
test results are descriptive; do not tune subsequent experiments against them.
The generated figures and metrics populate the local analytics page.
""")
    code('''for architecture in ("cnn", "efficientnet"):
    run("evaluate.py", "--model-dir", MODEL_DIRS[architecture], "--batch-size", str(BATCH_SIZE))
    for figure in ("accuracy", "val_accuracy", "loss", "val_loss", "confusion_matrix", "per_class_recall"):
        print(architecture, figure)
        display(Image(filename=f"reports/figures/{architecture}/{figure}.png"))
display(pd.read_csv("reports/metrics/comparison.csv"))
display(Image(filename="reports/figures/comparison.png"))
''')
    markdown("""## Optional independent rejection calibration

Prepare independent `data/ood_tune/` and `data/ood_audit/` collections, both
containing unrelated objects, non-leaf scenes and unsupported crops. Keep all
views of an object/leaf or capture session within one collection. Do not use
PlantVillage test images. Use at least 100 audit images. Mount Drive or upload
your own OOD archive and extract it into these directories. Alternatively use
`scripts/download_ood.py`: real UCI RGB leaves split by unsupported species and
CIFAR-10 source train/test objects, with exact and conservative near-duplicate
auditing. Public capture-session IDs are unavailable; document this limitation.
Both collections need `unsupported_leaves/` and `unrelated_objects/` subfolders.
Describe sources below. Without a passing calibration audit, the app displays real candidates
but keeps every assessment uncertain. That is intentional.
""")
    code('''RUN_CALIBRATION = True
DOWNLOAD_PUBLIC_OOD = True
OOD_DESCRIPTION = "UCI Leaf DOI 10.24432/C53C78 RGB-only with disjoint unsupported species and supported genera excluded; CIFAR-10 source train/tune and test/audit; exact and conservative near-duplicate removal; capture-session IDs unavailable. See data/ood_source/provenance.json."
if RUN_CALIBRATION:
    if DOWNLOAD_PUBLIC_OOD:
        run("scripts/download_ood.py")
    if not OOD_DESCRIPTION.strip():
        raise ValueError("Document the OOD collections first.")
    for architecture in ("cnn", "efficientnet"):
        run("calibrate.py", "--model-dir", MODEL_DIRS[architecture],
            "--ood-dir", "data/ood_tune", "--ood-audit-dir", "data/ood_audit",
            "--ood-description", OOD_DESCRIPTION, "--batch-size", str(BATCH_SIZE))
''')
    markdown("""## Package models and metadata for the local app

The package includes both `.keras` models, mappings, preprocessing contracts,
training history, metrics, figures, calibration (if generated) and catalogue
summary metadata. Raw training images and machine-specific manifests are omitted.
Extract the archive into the local project root, restart Streamlit and choose
`efficientnet_colab` or `cnn_colab` in the detection page's model selector. No
retraining is needed for inference. Retain dataset attribution and licenses.
""")
    code('''run("scripts/package_artifacts.py", "--output", "artifacts_leafguard.zip")
from google.colab import files
files.download("artifacts_leafguard.zip")
''')
    notebook = {"cells": cells, "metadata": {"colab": {"name": "LeafGuard AI - PlantVillage training", "provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.13"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(cells):
        cell["id"] = f"leafguard-{i:02d}"
    destination = ROOT / "notebooks/training_colab.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
