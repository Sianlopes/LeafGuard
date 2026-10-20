# LeafGuard AI

**Multi-Plant Disease Detection Using Deep Learning** — an academic Python project
with actual TensorFlow models, a leakage-aware data pipeline, evaluation,
uncertainty calibration, Grad-CAM and a five-page Streamlit application.

[GitHub](https://github.com/Sianlopes/LeafGuard) ·
[Open in Colab](https://colab.research.google.com/github/Sianlopes/LeafGuard/blob/main/notebooks/training_colab.ipynb)

Real trained model bundles, measured results and calibration artifacts accompany
this project. Raw training images and optional feature caches are not distributed.
Three attributed held-out example images are included for trying the app; their
source filenames and license are recorded in `data/examples/manifest.json`.
See `reports/project_report.md` for results and `reports/validation.md` for executed
checks. The application also handles absent artifacts with clear setup messages.

## Introduction and problem statement

Leaf appearance can help prioritize inspection, but pathogens, insects and
environmental stress can produce similar symptoms. The objective is to classify
the **defined PlantVillage plant-health categories**, compare learning from
scratch with transfer learning, and make limitations visible to the user. This
does not cover every plant species or establish a laboratory diagnosis.

## Setup

Use your installed Python **3.13**, then open a terminal in this project folder. Windows
requires a supported 64-bit Python and the TensorFlow runtime prerequisites.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell activation is disabled, run `.\.venv\Scripts\python.exe` in place
of `python`, and `.\.venv\Scripts\streamlit.exe` in place of `streamlit`.
On this PC the `.venv` environment is already installed. Double-click
`run_app.bat`, or run `.\.venv\Scripts\python.exe -m streamlit run app.py`
and open http://localhost:8501. Inference needs no dataset download or retraining.
The pins support Python 3.13: TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.3.2,
Streamlit 1.55.0 and Protobuf 6.33.5. Use a fresh environment rather than a
virtual environment that inherits unrelated packages. Older TensorFlow 2.18
pins do not support Python 3.13 and must not be mixed into this environment.

CPU training and inference work without CUDA. `--cpu` explicitly disables GPUs.
Native Windows TensorFlow 2.21 runs on CPU; use WSL2/Linux or Colab for NVIDIA
acceleration. On supported Linux/WSL installations, install
`python -m pip install "tensorflow[and-cuda]==2.21.0"` after the requirements.
Consult [TensorFlow installation documentation](https://www.tensorflow.org/install/pip)
for supported platform, driver and runtime combinations. Deterministic seeds
reduce variation but do not guarantee identical results across hardware versions.

## Dataset acquisition and provenance

Source: [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage).
The commonly cited collection contains approximately 54,306 original images,
14 crops and 38 categories; exact usable counts come from preparation.
This project uses **only the color variant**, not grayscale or segmented copies.

```bash
python -m src.dataset download
python -m src.dataset prepare
```

**Actual audit:** 54,305 color records were downloaded and decoded; 53,447 images
remain across all 38 categories after excluding 858 conflicting observations.
Authentic legacy label/filename aliases recover 2,025 physical-leaf matches;
11,790 source records still lack verified identities (11,500 among retained
images). Recovered metadata exposes 213 cross-source-split physical-leaf groups.
See `reports/source_audit.json` and `data/processed/metadata.json`.

The approved default `publisher` policy preserves source splits and audits
available physical-leaf IDs, exact decoded RGB hashes and conservative
near-duplicates. Unverified source-image identities are explicitly marked and
never described as physical-leaf IDs. **Complete physical-leaf separation cannot
be verified.** `--group-policy strict` continues to require complete verified
leaf metadata. No classes are silently removed. `python scripts/audit_source.py`
reproduces the metadata audit.

The downloader resolves a pinned source revision (`9e97599` by default), records
the complete commit in `data/raw/provenance.json`, checks the published archive
SHA-256, downloads official split lists and the physical-leaf map, and extracts
only `raw/color/`. No remote Python dataset loader is executed. The archive
download is approximately 2.18 GB (it also contains other variants); allow at
least 5 GB free space. Completed downloads are reused. An interrupted `.part`
download resumes when the server supports HTTP range requests; otherwise it restarts.

To choose explicit paths:

```bash
python -m src.dataset download --data-dir /path/to/plantvillage
python -m src.dataset prepare --data-dir /path/to/plantvillage --output data/processed
```

### Manual or offline acquisition

If network access is unavailable, download these files using another connected
computer from the **same revision** of the source repository:

* `data.zip`
* `leaf_grouping/leaf-map.json`
* `splits/color_train.txt` and `splits/color_test.txt`

For the default release, the archive SHA-256 is
`fba30c6a7965e49be94b47a62f8aff6cfb1c35c27f475f22092b56db41745e84`.
Extract the archive so a selected dataset root contains `raw/color/<class>/...`.
Convert verified/repaired source metadata offline using:

```bash
python -m src.dataset import-source --data-dir data/raw --leaf-map /path/to/leaf-map.json --train-list /path/to/color_train.txt --test-list /path/to/color_test.txt --archive /path/to/data.zip --group-policy publisher
```

The publisher policy also works offline with the verified official archive,
retaining the same documented physical-leaf limitation. Strict offline import
requires repaired physical-leaf metadata. Alternatively create `groups.csv` in that root using
verified physical-leaf IDs and split lists exported from your dataset manager.
The source lookup convention is implemented in `src.dataset.leaf_identifier`.
The downloader provides the automatic conversion when networking is available.

For an existing folder-based dataset, including the
[Kaggle alternative](https://www.kaggle.com/datasets/emmarex/plantdisease), labels
are discovered from the class directories; **do not assume that an alternative
contains the full 38-class collection**. A verified groups CSV is required:

```csv
path,label,leaf_id,split
raw/color/Apple___healthy/image_a.JPG,Apple___healthy,apple_leaf_001,train
raw/color/Apple___healthy/image_b.JPG,Apple___healthy,apple_leaf_001,train
raw/color/Potato___Early_blight/image_c.JPG,Potato___Early_blight,potato_leaf_002,test
```

This example documents the schema only; three images are insufficient to train.
Paths must resolve inside the dataset root. `label` defaults to the immediate
parent directory. `split` can be blank, train, validation, calibration or test.
`leaf_id` must identify the **physical leaf**, not the image UUID. If reliable
group IDs cannot be established, obtain the official metadata; the pipeline
refuses to imply that a random image split prevents same-leaf leakage.

```bash
python -m src.dataset prepare --data-dir /path/to/images --groups-csv /path/to/groups.csv
```

### Splits and leakage controls

Preparation decodes every supported image, skips and records unreadable images,
and discovers all supplied class labels. Available physical-leaf IDs, exact RGB
hashes and conservative near-duplicates form transitive groups. Near-duplicates
require 64-bit pHash distance <=2 and 32x32 RGB mean absolute difference <=3;
two such pairs were found in the real collection. Conflicting labels are
quarantined in `excluded.json`. Split conflicts retain the highest priority
observations (test, calibration, validation, train) and exclude lower-priority
matches. No official test image is reassigned to training. Other assignments
stay intact. A class lost entirely to corruption/conflict makes preparation fail.

When official train/test splits exist, missing validation and calibration sets
are drawn by class from **training leaf groups** (15% each of the training group
pool). The test assignments stay fixed. With no existing splits, 20% of groups
per class go to test, 15% to validation, 15% to calibration and the remainder to
training. Fractions apply to **groups**, so image counts vary. Every class must
appear in every split. Existing partial validation/calibration coverage must be
repaired rather than silently filled from held-out observations.

This conservative audit cannot detect every different photograph of the same
physical leaf. Source metadata quality limits leakage guarantees.

Outputs: `manifest.csv`, `class_names.json`, `summary.csv`, `metadata.json`,
`excluded.json`. The supported-plants UI uses these actual counts.
The manifest includes absolute paths: prepare again after moving raw data to a
different machine. Inference does **not** need raw data or a manifest.

## Methodology and architectures

All images are validated, EXIF-oriented, composited over white if transparent,
converted to RGB and bilinearly resized to 224 × 224. Both models receive
float32 values in **[0,255]**. CNN rescaling occurs inside the network;
[Keras EfficientNet preprocessing](https://keras.io/api/applications/efficientnet/)
is also embedded. There is no second external division by 255. The same Pillow
implementation is used in training and inference.

**Custom CNN (compact experiment):** an initial 16-channel stride-2 convolution,
32/64-channel separable convolutions, batch normalization, ReLU, max pooling
and 0.15 dropout in three blocks; a 96-channel 1x1 convolution; global average
pooling; 0.4 dropout; and a 38-way softmax classifier (13,798 parameters).
`--cnn-variant standard` retains the optional larger 32/64/128-channel baseline.

**EfficientNetB0:** ImageNet-pretrained backbone, initially frozen, followed by
global average pooling, 0.4 dropout and the class-sized softmax head. Optional
fine-tuning unfreezes the last 30 deeper layers except batch normalisation, with
a learning rate of 1e-5. If weights cannot be downloaded, the script stops with
instructions; `--weights none` explicitly runs an unpretrained experiment and is
recorded as such. It is never silently substituted for transfer learning.

Training-only augmentation uses flips, small rotations, zoom and contrast.
Validation, calibration, test and uploaded images are never augmented.
The completed CPU EfficientNet experiment freezes its real ImageNet backbone
throughout; fine-tuning is optional and was not used for that run.

## Training

Train each architecture independently:

```bash
python train.py --model cnn --epochs 15
python train.py --model efficientnet --epochs 15 --fine-tune-epochs 10
```

To reproduce the CPU EfficientNet experiment efficiently:

```bash
python -m pip install -r requirements-training.txt
python train.py --model efficientnet --epochs 15 --fine-tune-epochs 0 --cache-features openvino --feature-views 2 --threads 4 --cpu
```

OpenVINO accelerates frozen backbone forward passes only. The classification
head trains with TensorFlow/Keras, then its weights are copied into the complete
original `.keras` model. Original embeddings can be cached for all rows, while
augmented embeddings exist only for training rows. The cache is bound to the
manifest, backbone, seed and preprocessing; conversion must pass a numerical
comparison against TensorFlow. Model inference needs neither OpenVINO nor caches.
Omit `--cache-features` for the ordinary TensorFlow path. Both experiments use
balanced training-class weights. Existing bundles are protected; add
`--output models/experiment_name` for a new experiment.

Defaults in `config/config.yaml`: batch size 32, image size 224, Adam initial
learning rate 0.001, fine-tuning learning rate 0.00001, seed 42. CLI overrides:

```bash
python train.py --model efficientnet --epochs 15 --fine-tune-epochs 10 --batch-size 16 --image-size 224 --learning-rate 0.001 --fine-tune-learning-rate 0.00001 --unfreeze-layers 30 --seed 42 --cpu
```

Sparse categorical cross-entropy uses integer labels from the saved class list.
EarlyStopping, ModelCheckpoint and ReduceLROnPlateau monitor validation loss.
The best checkpoint is preserved across both training phases. Training history,
epoch CSV logs, run configuration, dataset fingerprint and inference metadata
are saved alongside `best.keras`. Existing experiments cannot be accidentally
overwritten: use `--output models/experiment_name` for a new run.

Full training on CPU can take many hours. Use Colab GPU for a college submission
if local hardware is insufficient. Do not report smoke-test performance as
plant-disease accuracy.

## Uncertainty calibration and unknown-image rejection

Softmax confidence is **not evidence that an image belongs to a supported crop**.
Before rejection validation passes, real model candidates are visible, but every
assessment is marked uncertain and no plant/disease diagnosis is accepted.

Collect two independent negative-image collections, each including unrelated
objects, non-leaf scenes and unsupported plant species. Use different capture
sessions for tuning and audit and keep views of the same object/leaf together.
Use real images; no synthetic OOD collection is silently used.
Use at least 100 independent audit images (more and broader data are preferable).

```bash
python calibrate.py --model-dir models/efficientnet --ood-dir data/ood_tune --ood-audit-dir data/ood_audit --ood-description "Document sources, species, unrelated objects, and independent capture sessions"
```

The reproducible real-data collection is prepared with
`python scripts/download_ood.py`. It uses [UCI Leaf](https://archive.ics.uci.edu/dataset/288/leaf)
(Silva and Maral, DOI 10.24432/C53C78, CC BY 4.0), RGB specimens only with
supported genera excluded, and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
Tuning and audit contain disjoint unsupported species and CIFAR source train
versus test images, with exact and conservative near-duplicates removed.
Tuning has 208 leaves + 250 objects; audit has 223 leaves + 250 objects.
Public sources lack capture-session IDs: session independence cannot be verified.
Use this limitation in `--ood-description`; full provenance is recorded in
`reports/ood_provenance.json`. Both OOD folders require `unsupported_leaves/`
and `unrelated_objects/` subdirectories so categories are tested separately.

Repeat with `--model-dir models/cnn` if demonstrating that network's rejection.
Calibration leaf groups are partitioned into three disjoint sets: temperature
fitting, threshold tuning and an independent audit. Mean embedding prototypes
are computed from **training images only**. Rejection requires both calibrated
confidence above a tuned threshold and cosine distance to the predicted class
prototype below a tuned cutoff. OOD images are used to tune the cutoff, with
separate OOD audit images used only for final checks. Exact pixel overlaps with
any prepared image, within OOD sets or between OOD sets are rejected. Physical
independence of external OOD capture sessions remains the collector's duty.

The default audit requires at least 100 in-distribution audit images, at least
100 OOD audit images, at least 30 accepted ID images, empirical accepted accuracy
≥90% and empirical OOD acceptance ≤5% overall and for each negative category
(at least 30 audit images per category). These are audit criteria, **not guaranteed
population performance**. Calibration records coverage, accepted accuracy, OOD
acceptance, ECE before/after scaling, temperature, sources and artifact hashes.
If tuning finds no feasible rule or the audit fails, `validated=false` and all
inferences remain uncertain. Collect a new audit set after any audit-informed
policy change; repeated tuning against the same audit invalidates independence.

The UI minimum-confidence slider can make this policy stricter. It cannot bypass
embedding-distance rejection or make an unaudited model appear validated.

## Evaluation and metrics

The completed EfficientNet CPU experiment achieved **96.78% test accuracy** and
**0.9604 macro F1** on 10,709 held-out images. Its separate rejection audit
accepted 2,047/2,100 ID images with 96.78% accepted accuracy and accepted
0/473 negative images. These are empirical results under the dataset and OOD
limitations documented above. The faster five-epoch CNN baseline comparison is
being completed separately; see the report for available measured results.

Choose the final model/hyperparameters using validation data, then evaluate the
frozen checkpoints on the held-out test split:

```bash
python evaluate.py --model-dir models/cnn
python evaluate.py --model-dir models/efficientnet
```

Outputs in `reports/metrics/` include JSON/CSV accuracy, macro precision/recall,
macro F1, weighted F1, a classification report, confusion-matrix CSV, per-class
precision/recall/F1/support and saved history. Training/validation accuracy and
loss figures, confusion heatmaps and per-class recall charts are written under
`reports/figures/<architecture>/`. A comparison chart and CSV are generated only
when both evaluations use the same manifest. Metrics describe un-rejected
closed-set classification; calibration audit statistics separately describe
selective acceptance and OOD rejection. Do not choose hyperparameters by test
scores. Evaluating both predefined models is a descriptive comparison, not
permission to keep optimizing against the test set.

## Run the application and prediction API

```bash
streamlit run app.py
python predict.py path/to/leaf.jpg --model-dir models/efficientnet
```

```python
from predict import predict_leaf
result = predict_leaf("path/to/leaf.jpg")
print(result["top_three"], result["uncertain"])
```

Home explains scope; Disease Detection supports image preview, explicit analysis,
real top-three probabilities, uncertainty, curated guidance, downloadable results
and optional Grad-CAM; Supported Plants is searchable; Model Performance reads
evaluation artifacts with interactive Plotly charts; About explains methodology.
Choose **Or try an example leaf** to demonstrate the actual inference flow
without supplying a photograph. No prediction appears until **Analyse Leaf**.

Select an installed model on the detection page or set `LEAFGUARD_MODEL_DIR` to
its full directory (PowerShell: `$env:LEAFGUARD_MODEL_DIR='models/cnn'`; Bash:
`export LEAFGUARD_MODEL_DIR=models/cnn`). Restart the process after replacing
artifacts to clear the reusable prediction API's model cache.

Grad-CAM differentiates the predicted class's pre-softmax score with respect to
actual final convolutional features (`cam_features` for CNN, `top_activation`
for EfficientNetB0). Zero-influence cases show an explanation, not a decorative
heatmap. Highlighted regions indicate model influence, not confirmed disease
locations.

JPEG and PNG content is checked rather than trusting the extension. Corrupt,
animated, excessive-size and very small uploads get actionable errors. Missing
model artifacts never produce substitute predictions. Bundles validate model
checksum, class count, input size and preprocessing contract on loading.

## Google Colab and deployment artifacts

1. Run `python scripts/package_source.py` locally.
2. Open `notebooks/training_colab.ipynb` in Colab and choose a GPU runtime.
3. The notebook clones `https://github.com/Sianlopes/LeafGuard.git` by default.
   To upload `leafguard_source.zip` instead, set `REPOSITORY_URL` to an empty
   string. Install requirements, download and inspect data, then
   train and evaluate both models using the provided cells.
4. Run real negative-image preparation and calibration, or supply your own
   independent collections with documented sources and limitations.
5. Run the notebook packaging cell and download `artifacts_leafguard.zip`.
6. Extract that archive **into this project's root**, preserving `models/`,
   `reports/` and `data/processed/` directories.
7. The notebook writes `models/efficientnet_colab/` and `models/cnn_colab/` to
   preserve the supplied CPU checkpoints. Verify the new directory contains `best.keras`, `class_names.json`,
   `inference_config.json`, `training_config.json` and `history.json` (plus
   `calibration.json` and `prototypes.npz` if calibrated). Restart Streamlit and
   select the new model in the detection page's model selector.

No retraining is needed for local inference. Keep metadata with its exact model;
never manually reorder class names. Packaged summary metadata supports the
catalogue without copying thousands of training images. Archived manifests are
intentionally omitted because their absolute training-machine paths are not
portable. To resume training/evaluation locally, download and prepare the same
dataset locally and train a matching experiment; exported metrics can be viewed
without raw data. Colab must use a runtime supporting the pinned TensorFlow
stack (Python 3.11–3.13). Reconnect/restart after dependency installation if Colab
requests it, then resume subsequent cells.

## Tests and verification

```bash
python -m pytest -q
python -m compileall -q app.py train.py evaluate.py predict.py calibrate.py src pages tests
```

Tests cover preprocessing values, invalid uploads, leaf grouping and duplicate
quarantine, source split preservation, repeatability, manifest tampering,
temperature scaling, impossible rejection policies, metrics, evaluation figures,
all five Streamlit startup states, and actual TensorFlow training/serialization/
inference/gradient paths. Tiny synthetic fixtures exist **only in isolated test
directories**; they are not PlantVillage replacements and never generate the
app's models, disease results or published metrics.

See `reports/validation.md` for what was actually executed in the build
environment. See `reports/project_report.md` for the academic report and measured
experimental results. Windows execution was verified locally; macOS/Linux and
execution inside a Google Colab runtime were not verified here.

## Limitations and future scope

PlantVillage backgrounds, lighting, label quality and class imbalance can bias
models. An unseen crop may receive high confidence. The supplied OOD audit
cannot prove universal unknown-image detection. Independent field validation,
expert-confirmed labels, broader geographic/camera coverage, severity estimation,
and more robust open-set methods are future work. Healthy classification does
not exclude latent disease. Guidance is educational and crop-specific; it never
prescribes pesticide concentrations or guaranteed treatments. Categories without
verified curated references show neutral advice (currently grape Isariopsis leaf spot).

## Source layout and GitHub

Entry points: `train.py`, `evaluate.py`, `calibrate.py`, `predict.py`, `app.py`.
Reusable modules live in `src/`; pages in `pages/`; configuration in `config/`;
curated reference information in `data/disease_info.json`; tests and Colab in
`tests/` and `notebooks/`. Generated raw data, models and metrics are ignored by
Git. Keep large artifacts in separate storage and include their checksum and
source attribution in submissions.

To create your own repository (publishing requires your GitHub destination):

```bash
git init
git add .
git commit -m "Implement LeafGuard AI training and inference project"
# Create a GitHub repository, then set its URL and push when ready.
```

## References

* Mohanty, Hughes & Salathé (2016), [Using Deep Learning for Image-Based Plant
  Disease Detection](https://doi.org/10.3389/fpls.2016.01419).
* Tan & Le (2019), [EfficientNet](https://arxiv.org/abs/1905.11946).
* Guo et al. (2017), [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).
* Selvaraju et al. (2017), [Grad-CAM](https://arxiv.org/abs/1610.02391).
* Disease guidance: linked University of Minnesota, University of Maryland,
  Penn State, Extension and UC IPM references in `data/disease_info.json`.

Dataset images and metadata retain their source licenses; the Hugging Face
repository lists CC BY-SA 3.0. No ownership of third-party data or ImageNet
weights is implied by this project.
