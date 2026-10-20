"""Prepare real unsupported leaves and unrelated objects, separately from training.

UCI Leaf (Silva & Maral, CC BY 4.0) supplies real leaf specimens. All Prunus,
Citrus, Vitis and other supported crop names are excluded. Species are split
between tuning and audit. CIFAR-10 tuning objects come from its source training
split and audit objects from its source test split. No generated images are used.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from collections import defaultdict
from io import BytesIO
import hashlib
import random
import zipfile
from PIL import Image
from src.common import ROOT, write_json, sha256
from src.dataset import fetch, write_rows, perceptual_fingerprint, HashTree
from src.preprocessing import load_image

CIFAR_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"
UCI_LEAF_SHA256 = "9d951760ac7609f5e824b2d996d8f1ab09d06febb67653a2abb84065ee379522"
CIFAR_FILES = {
    "train": ("train-00000-of-00001.parquet", "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883"),
    "test": ("test-00000-of-00001.parquet", "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de"),
}
SUPPORTED_NAMES = ("prunus", "citrus", "vitis", "malus", "blueberry", "vaccinium", "fragaria", "rubus", "solanum", "capsicum", "glycine", "cucurbita", "zea")


def download_cifar(directory):
    paths = {}
    for split, (filename, checksum) in CIFAR_FILES.items():
        paths[split] = fetch(f"https://huggingface.co/datasets/uoft-cs/cifar10/resolve/{CIFAR_REVISION}/plain_text/{filename}", Path(directory) / filename, checksum)
    return paths


def prepare_ood(directory=None, objects_per_split=250, seed=42):
    import numpy as np
    import pyarrow.parquet as pq
    source = Path(directory or ROOT / "data/ood_source")
    source.mkdir(parents=True, exist_ok=True)
    leaf_archive = fetch("https://archive.ics.uci.edu/static/public/288/leaf.zip", source / "uci_leaf.zip", UCI_LEAF_SHA256)
    cifar_paths = download_cifar(source)
    rng = random.Random(seed)
    records, pixels_seen = [], set()
    tree, thumbnails = HashTree(), []

    def save(image, destination, group, category, source_id):
        image = load_image(image)
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest in pixels_seen:
            return
        fingerprint, thumbnail = perceptual_fingerprint(image)
        for other in tree.query(fingerprint):
            if np.abs(thumbnail.astype(np.int16) - thumbnails[other].astype(np.int16)).mean() <= 3.:
                return
        tree.add(fingerprint, len(thumbnails))
        thumbnails.append(thumbnail)
        pixels_seen.add(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="PNG")
        records.append({"path": str(destination.relative_to(ROOT)), "split": group, "category": category, "source_id": source_id, "pixel_sha256": digest})

    with zipfile.ZipFile(leaf_archive) as archive:
        by_species = defaultdict(list)
        for member in archive.infolist():
            parts = member.filename.replace("\\", "/").split("/")
            if "RGB" not in parts or not member.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            rgb_index = parts.index("RGB")
            if len(parts) <= rgb_index + 2:
                continue
            species = parts[rgb_index + 1]
            if any(name in species.lower() for name in SUPPORTED_NAMES):
                continue
            by_species[species].append(member)
        species = sorted(by_species)
        rng.shuffle(species)
        if len(species) < 4:
            raise ValueError("The UCI archive did not contain enough unsupported RGB species.")
        tuning_species = set(species[:len(species) // 2])
        for name in species:
            group = "tune" if name in tuning_species else "audit"
            folder = ROOT / f"data/ood_{group}/unsupported_leaves" / name
            for member in by_species[name]:
                image = Image.open(BytesIO(archive.read(member)))
                save(image, folder / (Path(member.filename).stem + ".png"), group, "unsupported_leaves", member.filename)
    for source_split, group in (("train", "tune"), ("test", "audit")):
        table = pq.read_table(cifar_paths[source_split]).to_pydict()
        indices = list(range(len(table["label"])))
        rng.shuffle(indices)
        starting = len(records)
        for index in indices:
            value = table["img"][index] if "img" in table else table["image"][index]
            image = Image.open(BytesIO(value["bytes"]))
            label = int(table["label"][index])
            save(image, ROOT / f"data/ood_{group}/unrelated_objects/cifar_{label}/{source_split}_{index}.png", group, "unrelated_objects", f"cifar10:{CIFAR_REVISION}:{source_split}:{index}")
            if len(records) - starting >= objects_per_split:
                break
    write_rows(source / "manifest.csv", records)
    write_json(source / "provenance.json", {"seed": seed, "collections": [{"name": "UCI Leaf", "url": "https://archive.ics.uci.edu/dataset/288/leaf", "doi": "10.24432/C53C78", "license": "CC BY 4.0", "archive_sha256": sha256(leaf_archive), "tuning_species": sorted(tuning_species), "audit_species": sorted(set(species) - tuning_species)}, {"name": "CIFAR-10", "repository": "uoft-cs/cifar10", "revision": CIFAR_REVISION, "hashes": {split: sha256(path) for split, path in cifar_paths.items()}}], "images": len(records), "split_counts": {group: {category: sum(r["split"] == group and r["category"] == category for r in records) for category in ("unsupported_leaves", "unrelated_objects")} for group in ("tune", "audit")}, "separation": "Disjoint unsupported species; CIFAR source train versus test; RGB-only specimens; exact and conservative near-duplicate rejection across both collections.", "limitations": "Public source data do not provide capture-session IDs. These audits measure rejection on disjoint supplied images and species, not universal field OOD detection."})
    print(f"Prepared {len(records)} real OOD images. See {source / 'provenance.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-objects-only", action="store_true")
    parser.add_argument("--objects-per-split", type=int, default=250)
    args = parser.parse_args()
    if args.objects_per_split < 30:
        parser.error("At least 30 unrelated objects per split are required.")
    if args.download_objects_only:
        download_cifar(ROOT / "data/ood_source")
    else:
        prepare_ood(objects_per_split=args.objects_per_split)


if __name__ == "__main__":
    main()
