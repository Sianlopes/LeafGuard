"""Reusable predict_leaf(image) and command-line prediction."""
import argparse
import json
from src.inference import predict_leaf, ModelUnavailableError

__all__ = ["predict_leaf"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    try:
        print(json.dumps(predict_leaf(args.image, args.model_dir, args.threshold), indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Prediction error: {exc}\n")


if __name__ == "__main__":
    main()
