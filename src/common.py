"""Shared paths, configuration and artifact integrity helpers."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config(path=None):
    import yaml
    return yaml.safe_load(Path(path or ROOT / "config/config.yaml").read_text(encoding="utf-8"))


def parse_label(label):
    plant, separator, disease = label.partition("___")
    return plant.replace("_", " ").strip(), disease.replace("_", " ").strip() if separator else "Unknown category"


def model_directory():
    return Path(os.environ.get("LEAFGUARD_MODEL_DIR", ROOT / config()["model_dir"]))
