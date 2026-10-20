"""Download PlantVillage and build reproducible, duplicate-audited manifests.

Never execute remote dataset scripts. Download the versioned archive and
official metadata directly. Only the color variant is used.
"""
import argparse
import csv
import hashlib
import random
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
import numpy as np
from src.common import ROOT, read_json, write_json, sha256, config
from src.preprocessing import load_image

REPOSITORY = "mohanty/PlantVillage"
# Observed repository commit can be overridden and is recorded in provenance.
DEFAULT_REVISION = "9e97599"
ARCHIVE_SHA256 = "fba30c6a7965e49be94b47a62f8aff6cfb1c35c27f475f22092b56db41745e84"


def fetch(url, destination, expected_hash=None):
    import requests
    from tqdm import tqdm
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected_hash is None or sha256(destination) == expected_hash):
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            resumed = bool(offset and response.status_code == 206)
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Download server returned an incorrect resume offset.")
            offset = offset if resumed else 0
            with temporary.open("ab" if resumed else "wb") as handle, tqdm(total=offset + int(response.headers.get("content-length", 0)), initial=offset, unit="B", unit_scale=True, desc=destination.name, mininterval=10) as progress:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
                    progress.update(len(chunk))
        if expected_hash and sha256(temporary) != expected_hash:
            raise ValueError("Archive checksum mismatch. Delete the .part file and retry.")
        temporary.replace(destination)
    except Exception as exc:
        raise RuntimeError(f"Download failed: {url}. No data substituted. See README manual acquisition instructions. {exc}") from exc
    return destination


def safe_extract_color(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as handle:
        jobs = []
        for member in handle.infolist():
            parts = PurePosixPath(member.filename).parts
            if ".." in parts or "\\" in member.filename or PurePosixPath(member.filename).is_absolute():
                raise ValueError("Unsafe archive path.")
            if member.is_dir():
                continue
            if member.filename.startswith("color/"):
                parts = ("raw",) + parts
            elif not member.filename.startswith("raw/color/"):
                continue
            target = destination.joinpath(*parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive path escapes destination.")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            jobs.append((member, target))

        def extract(job):
            member, target = job
            with handle.open(member) as source, target.open("wb") as output:
                import shutil
                shutil.copyfileobj(source, output)
        with ThreadPoolExecutor(max_workers=6) as executor:
            for index, _ in enumerate(executor.map(extract, jobs), 1):
                if index % 5000 == 0 or index == len(jobs):
                    print(f"Extracted {index:,}/{len(jobs):,} color files", flush=True)


def normalized_leaf_map(mapping):
    normalized = defaultdict(list)
    for key, values in mapping.items():
        normalized[re.sub(r"\b0+(?=\d)", "", key.lower())].extend(values)
    return dict(normalized)


def leaf_identifier(filename, label, leaf_map):
    identifier = Path(filename).stem.replace("_final_masked", "").split("___")[-1].split("copy")[0].strip().lower()
    # The current filename prefix explicitly spells out Target Spot (TgS),
    # while its original same-class leaf spreadsheet uses Com.G_FL.
    if label == "Tomato___Target_Spot":
        identifier = identifier.replace("com.g_tgs_fl", "com.g_fl")
    identifier = re.sub(r"\b0+(?=\d)", "", identifier)
    candidates = leaf_map.get(identifier, [])
    source_label = "Apple_Frogeye Spot" if label == "Apple___Black_rot" else label
    matches = [str(item).replace(source_label + ":::", label + ":::", 1) for item in candidates if str(item).startswith(source_label + ":::")]
    if len(set(matches)) != 1:
        raise ValueError(f"Missing or ambiguous physical leaf metadata: {label}/{filename}. Supply a verified groups CSV.")
    return matches[0]


def source_record_group(filename, label):
    """An image-record identity, explicitly NOT a verified physical-leaf ID."""
    identifier = Path(filename).stem.replace("_final_masked", "").split("___")[-1]
    identifier = re.sub(r"\s*copy\s*\d*.*$", "", identifier, flags=re.IGNORECASE).strip().lower()
    return f"source-image:{label}:{identifier}"


def download_dataset(destination, revision=DEFAULT_REVISION, group_policy="strict"):
    import requests
    destination = Path(destination).resolve()
    api = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{revision}"
    try:
        response = requests.get(api, timeout=30)
        response.raise_for_status()
        commit = response.json()["sha"]
    except Exception as exc:
        raise RuntimeError("Cannot resolve dataset revision. See README for offline acquisition.") from exc
    base = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{commit}"
    meta = destination / "source"
    mapping = fetch(f"{base}/leaf_grouping/leaf-map.json", meta / commit / "leaf-map.json")
    leaf_map = normalized_leaf_map(read_json(mapping))
    rows = []
    unresolved = []
    for split in ("train", "test"):
        listing = fetch(f"{base}/splits/color_{split}.txt", meta / commit / f"color_{split}.txt")
        for relative in listing.read_text(encoding="utf-8").splitlines():
            relative = relative.strip()
            if not relative:
                continue
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative or not relative.startswith("raw/color/"):
                raise ValueError("Unsafe source split path.")
            label = path.parent.name
            try:
                leaf_id = leaf_identifier(path.name, label, leaf_map)
            except ValueError as exc:
                unresolved.append({"path": relative, "label": label, "source_split": split, "reason": str(exc)})
                rows.append({"path": relative, "label": label, "leaf_id": "", "split": split, "source_group": source_record_group(path.name, label), "group_basis": "source_image_unverified"})
                continue
            rows.append({"path": relative, "label": label, "leaf_id": leaf_id, "split": split, "source_group": leaf_id, "group_basis": "physical_leaf_metadata"})
    write_json(destination / "unresolved_source_records.json", unresolved)
    if unresolved and group_policy == "strict":
        write_rows(destination / "source_groups_to_review.csv", rows)
        raise ValueError(f"Published leaf map cannot uniquely resolve {len(unresolved):,} records. See {destination / 'unresolved_source_records.json'}. Obtain repaired physical-leaf metadata and use import-source or a verified groups.csv. No subset or filename-based leaf IDs were substituted; archive download not started.")
    archive = fetch(f"{base}/data.zip", destination / "data.zip", ARCHIVE_SHA256)
    safe_extract_color(archive, destination)
    write_rows(destination / "groups.csv", rows)
    write_json(destination / "provenance.json", {"repository": REPOSITORY, "revision": commit, "variant": "color", "archive_sha256": sha256(archive), "license": "CC BY-SA 3.0", "group_policy": group_policy, "unverified_group_records": len(unresolved), "groups_csv_sha256": sha256(destination / "groups.csv"), "limitation": "Publisher split preserved. Physical-leaf independence cannot be verified for records without matching leaf metadata; source-image identities are not physical-leaf IDs." if unresolved else None})
    print(f"Downloaded {len(rows):,} source records. Prepare them with python -m src.dataset prepare.")


def import_source(destination, leaf_map_path, train_list, test_list, archive=None, group_policy="strict", revision=DEFAULT_REVISION):
    """Convert manually downloaded official metadata without network access."""
    destination = Path(destination).resolve()
    mapping = normalized_leaf_map(read_json(leaf_map_path))
    rows = []
    unresolved = []
    if group_policy == "publisher" and not archive:
        raise ValueError("Publisher offline import requires --archive so the official image archive can be verified.")
    for split, listing in (("train", train_list), ("test", test_list)):
        for relative in Path(listing).read_text(encoding="utf-8").splitlines():
            relative = relative.strip()
            if not relative:
                continue
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative or not relative.startswith("raw/color/"):
                raise ValueError("Unsafe source split path.")
            label = path.parent.name
            try:
                leaf_id = leaf_identifier(path.name, label, mapping)
            except ValueError as exc:
                if group_policy == "strict":
                    raise
                leaf_id = ""
                unresolved.append({"path": relative, "label": label, "source_split": split, "reason": str(exc)})
            rows.append({"path": relative, "label": label, "leaf_id": leaf_id, "split": split, "source_group": leaf_id or source_record_group(path.name, label), "group_basis": "physical_leaf_metadata" if leaf_id else "source_image_unverified"})
    if archive:
        if sha256(archive) != ARCHIVE_SHA256:
            raise ValueError("Official archive checksum mismatch.")
        safe_extract_color(archive, destination)
    write_rows(destination / "groups.csv", rows)
    write_json(destination / "unresolved_source_records.json", unresolved)
    write_json(destination / "provenance.json", {"repository": REPOSITORY, "type": "manual import", "revision": revision, "variant": "color", "license": "CC BY-SA 3.0", "group_policy": group_policy, "unverified_group_records": len(unresolved), "groups_csv_sha256": sha256(destination / "groups.csv"), "archive_sha256": sha256(archive) if archive else None, "metadata_hashes": {str(Path(p).name): sha256(p) for p in (leaf_map_path, train_list, test_list)}, "limitation": "Physical-leaf identities are unavailable for unresolved source records; complete physical-leaf independence cannot be verified." if unresolved else None})
    return rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, a, b):
        self.parents[self.find(a)] = self.find(b)


def perceptual_fingerprint(image):
    """64-bit pHash and a color thumbnail for conservative duplicate confirmation."""
    from scipy.fft import dctn
    from PIL import Image
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
    low = dctn(gray, type=2, norm="ortho")[:8, :8].ravel()
    bits = low > np.median(low[1:])
    value = sum(int(bit) << i for i, bit in enumerate(bits))
    thumbnail = np.asarray(image.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.uint8)
    return value, thumbnail


class HashTree:
    """BK tree for close perceptual-hash candidates without quadratic pair scans."""
    def __init__(self):
        self.root = None

    def add(self, value, index):
        node = self.root
        if node is None:
            self.root = [value, [index], {}]
            return
        while True:
            distance = (node[0] ^ value).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value, radius=2):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (node[0] ^ value).bit_count()
            if distance <= radius:
                yield from node[1]
            pending.extend(child for edge, child in node[2].items() if distance - radius <= edge <= distance + radius)


def prepare_dataset(root, output, groups_csv=None, seed=42, validation_fraction=.15, calibration_fraction=.15, test_fraction=.2, group_policy="strict", workers=6):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError(f"Missing dataset: {root}. Run python -m src.dataset download.")
    if not 0 < validation_fraction < 1 or not 0 < calibration_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + calibration_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one.")
    groups_csv = Path(groups_csv or root / "groups.csv")
    if not groups_csv.exists():
        labels = sorted({p.parent.name for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and "___" in p.parent.name})
        raise ValueError(f"Found {len(labels)} class folders, but no groups.csv. Physical leaf IDs are required; filenames alone cannot guarantee leakage prevention. See README.")
    with groups_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"path", "leaf_id"}.issubset(rows[0]):
        raise ValueError("groups.csv requires path, leaf_id; optional label and split columns.")
    valid, excluded = [], []
    provenance = read_json(root / "provenance.json") if (root / "provenance.json").exists() else {"type": "local", "group_metadata": str(groups_csv)}
    publisher_allowed = (group_policy == "publisher" and provenance.get("repository") == REPOSITORY and provenance.get("group_policy") == "publisher" and provenance.get("groups_csv_sha256") == sha256(groups_csv))
    seen_paths = set()
    pending = []
    for row in rows:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("A dataset path escapes its root.")
        if str(path) in seen_paths:
            raise ValueError(f"Repeated path in groups.csv: {path}")
        seen_paths.add(str(path))
        label = row.get("label") or path.parent.name
        leaf_id = row["leaf_id"].strip()
        verified = bool(leaf_id and not leaf_id.lower().startswith(("unknown", "fallback")))
        if "___" not in label or (not verified and not publisher_allowed):
            raise ValueError(f"Invalid class or unverified leaf ID: {path}")
        if not verified and (not row.get("source_group", "").startswith("source-image:") or row.get("split") not in {"train", "test"}):
            raise ValueError(f"Missing publisher image-record identity: {path}")
        split = row.get("split", "").lower().strip()
        split = {"val": "validation", "valid": "validation"}.get(split, split)
        if split not in {"", "train", "validation", "calibration", "test"}:
            raise ValueError(f"Unrecognized split {split}")
        pending.append({"path": str(path), "label": label, "leaf_id": leaf_id if verified else "", "source_split": split, "source_group": leaf_id if verified else row["source_group"], "group_basis": "physical_leaf_metadata" if verified else "source_image_unverified"})

    def inspect(record):
        try:
            image = load_image(record["path"])
            fingerprint, thumbnail = perceptual_fingerprint(image)
            return {**record, "pixel_sha256": hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest(), "perceptual_hash": f"{fingerprint:016x}", "_thumbnail": thumbnail}, None
        except ValueError as exc:
            return None, {"path": record["path"], "reason": str(exc)}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, (record, error) in enumerate(executor.map(inspect, pending), 1):
            (valid if record is not None else excluded).append(record if record is not None else error)
            if index % 5000 == 0:
                print(f"Audited {index:,}/{len(pending):,} images", flush=True)
    if not valid:
        raise ValueError("No readable images found.")
    # Union transitive leaf IDs and exact decoded-pixel duplicates, even across labels.
    union = UnionFind(len(valid))
    keys = {}
    for i, row in enumerate(valid):
        for key in (("source_group", row["source_group"]), ("pixels", row["pixel_sha256"])):
            if key in keys:
                union.union(i, keys[key])
            keys[key] = i
    trees, near_duplicates = {}, []
    for i, row in enumerate(valid):
        tree = trees.setdefault(row["label"], HashTree())
        fingerprint = int(row["perceptual_hash"], 16)
        for other in tree.query(fingerprint):
            if valid[other]["pixel_sha256"] == row["pixel_sha256"]:
                continue
            difference = float(np.abs(row["_thumbnail"].astype(np.int16) - valid[other]["_thumbnail"].astype(np.int16)).mean())
            if difference <= 3.:
                union.union(i, other)
                near_duplicates.append({"path_a": valid[other]["path"], "path_b": row["path"], "hash_distance": (int(valid[other]["perceptual_hash"], 16) ^ fingerprint).bit_count(), "thumbnail_mean_absolute_difference": difference})
        tree.add(fingerprint, i)
    for row in valid:
        row.pop("_thumbnail", None)
    components = defaultdict(list)
    for i, row in enumerate(valid):
        components[union.find(i)].append(row)
    grouped = defaultdict(list)
    for group in components.values():
        labels = {r["label"] for r in group}
        assigned = {r["source_split"] for r in group if r["source_split"]}
        if len(labels) > 1:
            for row in group:
                excluded.append({"path": row["path"], "reason": "Conflicting labels within a leaf/duplicate group; quarantined."})
            continue
        if len(assigned) > 1:
            # Keep source held-out observations. Purge matching training images
            # rather than reassigning test data or retaining a leaking group.
            priority = {"train": 0, "validation": 1, "calibration": 2, "test": 3}
            retained_split = max(assigned, key=priority.get)
            for row in group:
                if row["source_split"] != retained_split:
                    excluded.append({"path": row["path"], "reason": f"Leaf/duplicate group also occurs in {retained_split}; lower-priority observation purged."})
            group = [r for r in group if r["source_split"] == retained_split]
        group_id = hashlib.sha256("|".join(sorted(r["pixel_sha256"] for r in group)).encode()).hexdigest()
        for row in group:
            row["group_id"] = group_id
        grouped[next(iter(labels))].append(group)
    rng = random.Random(seed)
    result = []
    all_labels = sorted({r.get("label") or Path(r["path"]).parent.name for r in rows})
    if set(grouped) != set(all_labels):
        raise ValueError("An entire class was lost to corruption or conflicting metadata. Repair the data before training.")
    existing = {r["source_split"] for groups in grouped.values() for g in groups for r in g if r["source_split"]}
    for label, groups in sorted(grouped.items()):
        rng.shuffle(groups)
        train_pool = [g for g in groups if g[0]["source_split"] in {"", "train"}]
        retained = [g for g in groups if g[0]["source_split"] not in {"", "train"}]
        if "test" in existing and not any(g[0]["source_split"] == "test" for g in retained):
            raise ValueError(f"No existing test groups for {label}.")
        needed = [s for s in ("test", "validation", "calibration") if s not in existing]
        if len(train_pool) < len(needed) + 1:
            raise ValueError(f"Too few independent leaf groups in {label}; need at least {len(needed)+1} training-pool groups.")
        total = len(train_pool)
        fractions = {"test": test_fraction, "validation": validation_fraction, "calibration": calibration_fraction}
        for index, split in enumerate(needed):
            n = min(max(1, round(total * fractions[split])), len(train_pool) - (len(needed) - index))
            for group in train_pool[:n]:
                for row in group:
                    row["split"] = split
            retained.extend(train_pool[:n])
            train_pool = train_pool[n:]
        for group in train_pool:
            for row in group:
                row["split"] = "train"
        for group in retained:
            for row in group:
                row.setdefault("split", row["source_split"])
        result.extend(row for group in train_pool + retained for row in group)
    classes = sorted(grouped)
    for row in result:
        row["class_index"] = classes.index(row["label"])
    for label in classes:
        if {r["split"] for r in result if r["label"] == label} != {"train", "validation", "calibration", "test"}:
            raise ValueError(f"Every class must appear in all four splits: {label}")
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "manifest.csv", result)
    write_json(output / "class_names.json", classes)
    if output == (ROOT / "data/processed").resolve():
        write_json(ROOT / "data/class_names.json", classes)
    write_json(output / "excluded.json", excluded)
    write_rows(output / "near_duplicates.csv", near_duplicates)
    summary = [{"label": label, **{split: sum(r["label"] == label and r["split"] == split for r in result) for split in ("train", "validation", "calibration", "test")}} for label in classes]
    write_rows(output / "summary.csv", summary)
    unverified = sum(r["group_basis"] == "source_image_unverified" for r in result)
    write_json(output / "metadata.json", {"seed": seed, "classes": classes, "images": len(result), "excluded": len(excluded), "groups": len({r["group_id"] for r in result}), "manifest_sha256": sha256(output / "manifest.csv"), "source": provenance, "group_policy": group_policy, "unverified_physical_leaf_records": unverified, "complete_physical_leaf_verification": unverified == 0, "near_duplicate_pairs": len(near_duplicates), "leakage_checks": ["available physical leaf IDs", "decoded RGB pixel duplicates", "pHash distance <=2 confirmed by color-thumbnail difference <=3", "source split conflicts quarantined"], "limitations": "Complete physical-leaf separation cannot be verified for source-image groups without physical-leaf metadata. Perceptual hashing detects near-identical views, not every view of the same leaf."})
    print(f"Prepared {len(result):,} images, {len(classes)} classes; {len(excluded)} excluded.\n" + str(output / "summary.csv"))
    return result


def tf_dataset(rows, size=224, batch_size=32, training=False, seed=42):
    import tensorflow as tf
    from src.preprocessing import preprocess_image
    paths = [r["path"] for r in rows]
    labels = [int(r["class_index"]) for r in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        def read(value):
            return preprocess_image(value.numpy().decode("utf-8"), size)
        image = tf.py_function(read, [path], Tout=tf.float32)
        image.set_shape((size, size, 3))
        return image, label

    # Bound buffering on 16 GB laptops. AUTOTUNE across multiple full-resolution
    # training jobs can exhaust physical RAM and make paging dominate execution.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    options.threading.max_intra_op_parallelism = 1
    return dataset.map(decode, num_parallel_calls=4).batch(batch_size).prefetch(1).with_options(options)


def load_manifest(directory):
    directory = Path(directory)
    path = directory / "manifest.csv"
    if not path.exists():
        raise ValueError("No prepared dataset. Run python -m src.dataset prepare first.")
    metadata = read_json(directory / "metadata.json")
    if sha256(path) != metadata["manifest_sha256"]:
        raise ValueError("Manifest changed after preparation; prepare it again.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle)), read_json(directory / "class_names.json"), metadata


def main():
    defaults = config()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--data-dir", default=defaults["dataset_path"])
    download.add_argument("--revision", default=DEFAULT_REVISION)
    download.add_argument("--group-policy", choices=["strict", "publisher"], default=defaults.get("group_policy", "strict"))
    manual = sub.add_parser("import-source")
    manual.add_argument("--data-dir", default=defaults["dataset_path"])
    manual.add_argument("--leaf-map", required=True)
    manual.add_argument("--train-list", required=True)
    manual.add_argument("--test-list", required=True)
    manual.add_argument("--archive")
    manual.add_argument("--group-policy", choices=["strict", "publisher"], default="strict")
    manual.add_argument("--revision", default=DEFAULT_REVISION)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--data-dir", default=defaults["dataset_path"])
    prepare.add_argument("--output", default=defaults["processed_path"])
    prepare.add_argument("--groups-csv")
    prepare.add_argument("--seed", type=int, default=defaults["seed"])
    prepare.add_argument("--group-policy", choices=["strict", "publisher"], default=defaults.get("group_policy", "strict"))
    prepare.add_argument("--workers", type=int, default=6)
    for name in ("validation", "calibration", "test"):
        prepare.add_argument(f"--{name}-fraction", type=float, default=defaults[f"{name}_fraction"])
    args = parser.parse_args()
    try:
        if args.command == "download":
            download_dataset(args.data_dir, args.revision, args.group_policy)
        elif args.command == "import-source":
            imported = import_source(args.data_dir, args.leaf_map, args.train_list, args.test_list, args.archive, args.group_policy, args.revision)
            print(f"Imported {len(imported)} metadata records. Run the prepare command next.")
        else:
            prepare_dataset(args.data_dir, args.output, args.groups_csv, args.seed, args.validation_fraction, args.calibration_fraction, args.test_fraction, args.group_policy, args.workers)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Dataset error: {exc}\n")


if __name__ == "__main__":
    main()
