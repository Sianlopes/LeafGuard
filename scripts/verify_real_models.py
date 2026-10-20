"""Verify the installed real checkpoints against held-out images and gradients."""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.common import ROOT, write_json
from src.dataset import load_manifest
from src.preprocessing import preprocess_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["cnn", "efficientnet"], default=["cnn", "efficientnet"])
    args = parser.parse_args()
    from src.training_utils import set_seed
    set_seed(42, True, 4)
    from src.inference import Predictor
    from src.gradcam import gradcam
    from src.feature_cache import features_for_rows, predict_from_features
    rows, classes, metadata = load_manifest(ROOT / "data/processed")
    selected = [next(row for row in rows if row["split"] == "test" and row["label"] == label) for label in classes]
    pixels = np.stack([preprocess_image(row["path"], 224) for row in selected])
    report = {"manifest_sha256": metadata["manifest_sha256"], "samples": len(selected), "sampling": "First manifest test image per class; verification only, no policy changes", "models": {}}
    for name in args.models:
        predictor = Predictor(ROOT / "models" / name)
        raw = np.asarray(predictor.model(pixels, training=False))
        assert raw.shape == (len(classes), len(classes)) and np.isfinite(raw).all()
        assert np.allclose(raw.sum(axis=1), 1, atol=1e-4)
        details = {"model_sha256": predictor.settings["model_sha256"], "parameters": predictor.model.count_params(), "full_tensorflow_output_valid": True}
        cached = features_for_rows(predictor, selected)
        if cached is not None:
            accelerated = predict_from_features(predictor, cached)
            agreement = float((raw.argmax(axis=1) == accelerated.argmax(axis=1)).mean())
            maximum = float(np.abs(raw - accelerated).max())
            assert agreement == 1 and maximum < .002, (agreement, maximum)
            details["cache_agreement_on_real_test_images"] = {"top_one_agreement": agreement, "maximum_probability_absolute_error": maximum}
        result = predictor.predict(selected[0]["path"])
        assert len(result["top_three"]) == 3 and result["calibrated"]
        details["first_test_example"] = {"source_filename": Path(selected[0]["path"]).name, "actual": selected[0]["label"], "result": result}
        try:
            overlay, heat = gradcam(predictor, selected[0]["path"])
            assert np.isfinite(heat).all() and heat.max() > 0
            output = ROOT / "reports/figures" / name / "gradcam_example.png"
            overlay.save(output)
            details["gradcam"] = {"positive_influence": True, "file": str(output.relative_to(ROOT)), "source": "PlantVillage, Mohanty/Hughes/Salathe; CC BY-SA 3.0", "filename": Path(selected[0]["path"]).name}
        except ValueError as exc:
            if "No positive class influence" not in str(exc):
                raise
            details["gradcam"] = {"positive_influence": False, "explanation": str(exc)}
        ood_examples = []
        for category in ("unsupported_leaves", "unrelated_objects"):
            paths = sorted((ROOT / "data/ood_audit" / category).rglob("*.png"))[:5]
            for path in paths:
                outcome = predictor.predict(path)
                ood_examples.append({"category": category, "source_filename": path.name, "uncertain": outcome["uncertain"], "confidence": outcome["confidence"]})
        details["ood_examples"] = ood_examples
        details["calibration_validated"] = predictor.calibration["validated"]
        report["models"][name] = details
        print(f"{name}: native TensorFlow inference, calibration load and Grad-CAM verified.", flush=True)
    write_json(ROOT / "reports/integration.json", report)


if __name__ == "__main__":
    main()
