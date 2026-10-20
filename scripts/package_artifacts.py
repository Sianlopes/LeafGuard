"""Package trained models, evaluation reports and UI dataset metadata; no raw data."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts_leafguard.zip")
    args = parser.parse_args()
    checkpoints = list((ROOT / "models").glob("*/best.keras"))
    if not checkpoints:
        parser.error("No trained model exists; train at least one architecture first.")
    files = []
    for directory in (ROOT / "models", ROOT / "reports/metrics", ROOT / "reports/figures"):
        files.extend(p for p in directory.rglob("*") if p.is_file() and p.suffix in {".keras", ".json", ".csv", ".png", ".npz"})
    for name in ("summary.csv", "metadata.json", "class_names.json"):
        path = ROOT / "data/processed" / name
        if path.exists():
            files.append(path)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"Packaged {len(set(files))} artifact files into {args.output}. Extract into your local project root, then restart Streamlit.")


if __name__ == "__main__":
    main()
