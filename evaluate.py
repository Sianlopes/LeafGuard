"""Evaluate a frozen validation-selected checkpoint on the held-out test set."""
import argparse
from pathlib import Path
from src.common import read_json, sha256
from src.dataset import load_manifest, tf_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/efficientnet")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    try:
        from src.training_utils import set_seed
        set_seed(42, args.cpu, args.threads)
        from src.inference import Predictor
        from src.evaluation import save_evaluation
        predictor = Predictor(args.model_dir)
        rows, classes, metadata = load_manifest(args.data_dir)
        if classes != predictor.classes or metadata["manifest_sha256"] != predictor.settings["manifest_sha256"]:
            raise ValueError("Evaluation dataset/class mapping does not match the trained experiment.")
        test_rows = [r for r in rows if r["split"] == "test"]
        if not test_rows:
            raise ValueError("No held-out test images.")
        from src.feature_cache import features_for_rows, predict_from_features
        cached = features_for_rows(predictor, test_rows)
        if cached is None:
            dataset = tf_dataset(test_rows, predictor.settings["image_size"], args.batch_size)
            probabilities = predictor.model.predict(dataset, verbose=1)
            backend = "TensorFlow full model"
        else:
            probabilities = predict_from_features(predictor, cached)
            backend = "Frozen cached embeddings plus exact TensorFlow classifier; see feature_cache.json for numerical agreement audit"
        history = read_json(Path(args.model_dir) / "history.json")
        metrics = save_evaluation(predictor.settings["architecture"], classes, [int(r["class_index"]) for r in test_rows], probabilities, history, args.output, {"manifest_sha256": metadata["manifest_sha256"], "model_sha256": sha256(Path(args.model_dir) / "best.keras"), "checkpoint_selection": "minimum validation loss; test not used for selection", "evaluation_backend": backend, "complete_physical_leaf_verification": metadata.get("complete_physical_leaf_verification", True), "split_limitation": metadata.get("limitations")})
        print(metrics)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
