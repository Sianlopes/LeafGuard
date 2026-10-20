"""Create a small, reviewable source archive for upload into Google Colab."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="leafguard_source.zip")
    args = parser.parse_args()
    files = []
    for pattern in ("*.py", "*.txt", "*.md", "*.bat", ".gitignore", ".gitattributes", ".python-version", "pytest.ini", "src/*.py", "scripts/*.py", "pages/*.py", "tests/*.py", "config/*.yaml", ".streamlit/*.toml", ".github/workflows/*.yml", "data/disease_info.json", "notebooks/*.ipynb", "reports/*.md", "reports/source_audit.json", "reports/ood_provenance.json", "reports/preparation_audit.json", "reports/integration.json"):
        files.extend(ROOT.glob(pattern))
    files.extend(path for path in (ROOT / "data/examples").glob("*") if path.is_file())
    files.extend(path for path in (ROOT / "assets").glob("*") if path.is_file())
    output = Path(args.output).resolve()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if path != output:
                archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"Source package: {output} ({len(set(files))} files)")


if __name__ == "__main__":
    main()
