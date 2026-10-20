# Execution and validation record

Status date: **20 October 2026**, Asia/Calcutta.

## Runtime

The project uses its own Python 3.13.7 `.venv` with the complete pinned
requirements installed. `pip check` passed. TensorFlow 2.21.0, Keras 3.15.1,
NumPy 2.3.2, Streamlit 1.55.0 and Protobuf 6.33.5 import together. OpenVINO
2025.4.1 is an optional training accelerator, separate from app requirements.
The local machine has an Intel Core i5-12450H, approximately 16 GB RAM and an
NVIDIA RTX 3050 laptop GPU. Native Windows TensorFlow runs on CPU; WSL and
GPU training were not installed or used for this run.

## Real data preparation completed

* Verified and extracted the official approximately 2.18 GB PlantVillage archive.
* Decoded all 54,305 color records across all 38 categories and 14 crops.
* Authentic legacy aliases recovered 2,025 metadata matches; 11,790 source
  records still lack verified physical-leaf IDs.
* The user explicitly approved publisher splits plus duplicate auditing and
  disclosure that complete physical-leaf separation cannot be verified.
* Recovered metadata exposes 213 cross-source-split physical-leaf groups.
  Removed 858 matching lower-priority observations, retaining every publisher
  test image and every category. Two conservative near-duplicate pairs were found.
* Final counts: 29,941 train, 6,397 validation, 6,400 calibration and 10,709 test.
* Independent manifest checks found zero cross-split overlap for available leaf
  IDs, source groups, transitive group IDs and decoded RGB hashes.
* 11,500 retained images have unverified physical-leaf identity. These checks do
  not establish complete physical-leaf independence.
* Downloaded and prepared 931 real negative images from UCI Leaf and CIFAR-10:
  458 for tuning and 473 for audit, with unsupported species disjoint and
  conservative exact/near-duplicate auditing. Capture-session IDs are unavailable.

See `preparation_audit.json`, `source_audit.json`, `ood_provenance.json` and
`data/processed/metadata.json` for counts, checksums and limitations.

## Software checks executed

The final clean-runtime regression run passed **all 37 tests** in 59.78 seconds,
including actual CNN optimizer
updates, a complete training/evaluation CLI round trip, EfficientNet graph and
fine-tuning gradients, native Keras save/load, calibration and Grad-CAM paths,
evaluation figures and comparison export, and all five Streamlit pages.
The run used the complete installed environment after the real EfficientNet
artifacts were installed. One Keras input-structure warning came from a small
test graph; its inference and gradient checks passed. Separate real-artifact
verification and the example-to-analysis app interaction also passed.

Synthetic test images, small fixture models and fixture metrics live only in
ignored test directories; they never become app artifacts or experiment scores.
Every generated Colab code cell parses, and the checked-in notebook has no saved
outputs. Source and artifact packaging scripts are included.

## Real model execution

Official ImageNet EfficientNetB0 weights were downloaded. OpenVINO float32
backbone conversion was checked against TensorFlow on identical probe pixels:
maximum absolute embedding error 0.000007391, mean 0.000000423. Allowed comparison
tolerances were rtol=0.002 and atol=0.002. Training and evaluation record whether
frozen cached features are used; the full app model remains native TensorFlow.

The initial simultaneous CPU runs were interrupted before either completed a
checkpoint. Completed feature-cache batches were preserved. Less than 1 GB of
physical RAM was available during that run. Input buffering was bounded, and
training resumed with one job at a time. No partial CNN checkpoint or random
ImageNet classification head is installed as a plant-disease model.

EfficientNet completed all 15 classification-head epochs with its ImageNet
backbone frozen. The best validation-loss checkpoint achieved 96.7784% accuracy
and 0.960406 macro F1 on all 10,709 held-out test images. Evaluation reused verified
fixed embeddings with the exact saved TensorFlow classifier. Native full-model
TensorFlow checks on 38 real held-out images (one per class) matched every top
prediction; maximum probability difference was 0.000001550.

Its rejection audit passed: 2,047 of 2,100 ID audit images were accepted
(97.4762% coverage), with 96.7758% accepted accuracy. Zero of 473 negative audit
images were accepted, including 223 unsupported leaves and 250 unrelated objects.
These empirical results do not establish universal rejection or field accuracy.
Real native-model inference, prototype/calibration integrity, Grad-CAM generation
and the complete example-selection/Analyse Leaf/Grad-CAM Streamlit flow passed.
See `integration.json` and `app_verification.json`.

The user requested faster completion. The CNN baseline therefore uses a fixed
five-epoch budget; training and comparison remain in progress. The complete
functional EfficientNet app and its actual artifacts were published to
https://github.com/Sianlopes/LeafGuard at commit `c86bf66` while the baseline runs.

## App review and external execution limits

All five pages passed Streamlit AppTest navigation checks. A local server starts
at http://localhost:8501. Earlier health verification returned HTTP 200 `ok`;
the final trained app will be checked again.

Windows Computer Use could list Chrome, but visual inspection ended because the
helper could not determine the current browser URL confidently enough to enforce
its policy. Desktop interaction stopped at that point. No visual-browser QA is
claimed. The separate browser connector also had no available browser provider.

Ubuntu dependency installation, all regression tests and compilation passed in
GitHub Actions run 39397568139 after publication. A fresh remote checkout also
preserved every model/class/prototype checksum and included all 38 classes,
without raw data or feature caches. Windows and Ubuntu checks are verified.

Execution inside Google Colab, macOS, GPU training and independent field
validation have not been performed. The output-free Colab notebook implements
source acquisition, full data preparation, both training runs, evaluation,
optional real negative calibration and portable artifact packaging; local
inference does not require a Colab run.
